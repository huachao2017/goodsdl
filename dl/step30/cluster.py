import logging
import time
import os
from urllib import request, parse
import socket
import shutil
import django
from django.db.models import Q
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import TrainAction, TrainTask, ClusterStructure, ClusterEvalData, ClusterEvalStep, ClusterSampleScore, ClusterUpcScore
import tensorflow as tf


def _run_cluster(task, precision, labels_to_names):
    """
    3.3.1、计算单样本聚类打分，算法：最近3次checkpoint的score，按60%，30%，10%，加权平均。（TODO：这部分可以根据map和category_ap的数据自学习）
    3.3.2、聚类打分算法：取A->B或B->A单样本最高值作为聚类结果
    3.3.3、聚类：设定50%为阀值进行聚类连接，记录到cluster_structure表中，修改目录存储结构，创建子训练任务
    3.3.4、收尾：终止训练进程，当前task设为0未开始，新建一个拷贝的task，重启次数+1，修订分类总数，map清零。

    :param task:
    :param precision:
    :return:
    """
    logging.info('begin cluster:{}'.format(task.pk))

    # 3.3.1、计算单样本聚类打分，算法：最近3次checkpoint的score，按60%，30%，10%，加权平均。（TODO：这部分可以根据map和category_ap的数据自学习）
    use_steps = []
    db_steps = ClusterEvalStep.object.filter(train_task_id=task.pk).order_by('-checkpoint_step')
    for step in db_steps:
        if step in use_steps:
            continue
        use_steps.append(step)
        if len(use_steps) == 3:
            break

    sample_scores = {}
    # tag:step:{groundtruth_label:nn,detection_label:nn,score:ff}
    eval_datas = ClusterEvalData.objects.filter(train_task_id=task.pk).filter(
        Q(checkpoint_step=use_steps[0]) | Q(checkpoint_step=use_steps[1]) | Q(checkpoint_step=use_steps[2]))
    for eval_data in eval_datas:
        if eval_data.sample_serial not in sample_scores:
            sample_scores[eval_data.sample_serial] = {}
        sample_scores[eval_data.sample_serial][eval_data.checkpoint_step] = {
            'groundtruth_label': eval_data.groundtruth_label,
            'detection_label': eval_data.detection_label,
            'score': eval_data.score,
        }

    for sample_serial in sample_scores:
        one_sample = sample_scores[sample_serial]
        detections = {}
        for checkpoint_step in one_sample:
            weight = 0.0
            if checkpoint_step == use_steps[0]:
                weight = .6
            elif checkpoint_step == use_steps[1]:
                weight = .3
            elif checkpoint_step == use_steps[2]:
                weight = .1

            if one_sample[checkpoint_step]['detection_label'] in detections:
                detections[one_sample[checkpoint_step]['detection_label']] = detections[one_sample[checkpoint_step][
                    'detection_label']] + weight * one_sample[checkpoint_step]['score']
            else:
                detections[one_sample[checkpoint_step]['detection_label']] = weight * one_sample[checkpoint_step][
                    'score']

        for detection in detections:
            ClusterSampleScore.objects.create(
                train_task_id=task.pk,
                sample_serial=sample_serial,
                upc_1=labels_to_names[one_sample['groundtruth_label']],
                upc_2=labels_to_names[detection],
                score=detections[detection]
            )

    # 3.3.2、聚类打分算法：取A->B或B->A单样本最高值作为聚类结果
    sample_scores = ClusterSampleScore.objects.filter(train_task_id=task.pk)
    upc_scores = {}
    for sample_score in sample_scores:
        if sample_score.upc_1 in upc_scores:
            if sample_score.upc_2 in upc_scores[sample_score.upc_1]:
                if upc_scores[sample_score.upc_1][sample_score.upc_2] < sample_score.score:
                    upc_scores[sample_score.upc_1][sample_score.upc_2] = sample_score.score
            else:
                upc_scores[sample_score.upc_1][sample_score.upc_2] = sample_score.score
        else:
            upc_scores[sample_score.upc_1] = {sample_score.upc_2: sample_score.score}

    for upc_1 in upc_scores:
        for upc_2 in upc_scores[upc_1]:
            # 防止重复
            duplicate = ClusterUpcScore.objects.filter(train_task_id=task.pk).filter(upc_1=upc_2).filter(upc_2=upc_1)[
                        :1]
            if len(duplicate) > 0:
                if duplicate[0].score < upc_scores[upc_1][upc_2]:
                    duplicate[0].score = upc_scores[upc_1][upc_2]
                    duplicate[0].save()
            else:
                ClusterUpcScore.objects.create(
                    train_task_id=task.pk,
                    upc_1=upc_1,
                    upc_2=upc_2,
                    score=upc_scores[upc_1][upc_2]
                )

    # 3.3.3.1、聚类：设定50%为阀值进行聚类连接
    upc_scores = ClusterUpcScore.objects.filter(train_task_id=task.pk)
    source_clusters = {}  # {A:[B,C,D],B:[C,E],C:[F],E:[F],G:[A,H]}
    for upc_score in upc_scores:
        if upc_score.score < 0.5:
            continue
        if upc_score.upc_1 in source_clusters:
            if upc_score.upc_2 in source_clusters[upc_score.upc_1]:
                continue
            else:
                source_clusters[upc_score.upc_1].append(upc_score.upc_2)
        else:
            source_clusters[upc_score.upc_1] = [upc_score.upc_1]

    sorted_cluster = {}  # {A:[B,C,D,G,H],B:[C,E],C:[F],E:[F]}
    for key in source_clusters:
        ones = [key]
        for one in source_clusters[key]:
            ones.append(one)
        sorted_ones = sorted(ones)
        f_upc = sorted_ones[0]
        sorted_ones.remove(f_upc)
        if f_upc in sorted_cluster:
            for one in sorted_ones:
                if one not in sorted_cluster[f_upc]:
                    sorted_cluster[f_upc].append(one)
        else:
            sorted_cluster[f_upc] = sorted_ones

    def inner_find(f_upc, sorted_cluster):
        ones = sorted_cluster[f_upc]
        ret = ones.copy()
        for one in ones:
            finds = inner_find(one, sorted_cluster)
            for find_one in finds:
                if find_one not in ret:
                    ret.append(find_one)

        return ret

    solved_cluster = {}  # {A:[B,C,D,E,F,G,H]}
    for f_upc in sorted_cluster:
        solved_cluster[f_upc] = inner_find(f_upc, sorted_cluster)

    # 3.3.3.2、记录到cluster_structure表中
    for father in solved_cluster:
        f_structure = ClusterStructure.objects.filter(upc=father).order_by('id')[:1]
        for upc in solved_cluster[father]:
            c_structure = ClusterStructure.objects.filter(upc=upc).order_by('id')[:1]
            c_structure[0].f_upc = f_structure[0].upc
            c_structure[0].save()

    # 3.3.3.3、修改目录存储结构
    tmp_dir = os.path.join(task.dataset_dir, 'tmp')
    if not tf.gfile.Exists(tmp_dir):
        tf.gfile.MakeDirs(tmp_dir)
    for father in solved_cluster:
        father_dir = os.path.join(task.dataset_dir,father)
        father_tmp_dir = os.path.join(tmp_dir,father)
        shutil.move(father_dir,tmp_dir)
        tf.gfile.MakeDirs(father_dir)
        shutil.move(father_tmp_dir,father_dir)
        for upc in solved_cluster[father]:
            upc_dir = os.path.join(task.dataset_dir,upc)
            shutil.move(upc_dir,father_dir)
    os.remove(tmp_dir)

    # 3.3.3.4、创建子训练任务
    for father in solved_cluster:
        father_dir = os.path.join(task.dataset_dir,father)
        TrainTask.objects.create(
            dataset_dir=father_dir,
        )

    # 3.3.4、收尾：终止训练进程，当前task设为0未开始，新建一个拷贝的task，重启次数+1，修订分类总数，map清零。

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def _run_export(domain, task, precision):
    if domain is None:
        domain = get_host_ip()
    train = TrainAction.objects.filter(action='T30').filter(
        dataset_dir=task.dataset_dir).order_by('-update_time')[:1]
    train_data = parse.urlencode([
        ('train_action', train.pk),
        ('model_name', train.model_name),
        ('precision', precision),
    ])
    req = request.Request('http://{}/api/exportaction/'.format(domain))
    req.add_header('Origin', 'http://{}/api/'.format(domain))
    req.add_header('User-Agent',
                   'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
    # print(req.full_url)
    with request.urlopen(req, data=train_data.encode('utf-8')) as f:
        print('_run_export status:', f.status, f.reason)
        if f.status != 201:
            raise ValueError('export error')
        else:
            task.state = 2
            task.m_ap = precision
            task.save()

