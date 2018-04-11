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


def _run_cluster(task, precision, labels_to_names, train_dir):
    """
    3.3.1、计算单样本聚类打分，算法：最近3次checkpoint的score，按50%，30%，20%，加权平均。（TODO：这部分可以根据map和category_ap的数据自学习）
    3.3.2、聚类打分算法：取A->B或B->A单样本最高值作为聚类结果
    3.3.3、聚类：设定45%为阀值进行聚类连接，判断是否聚类后分类数限制，修改目录存储结构，记录到cluster_structure表中
    3.3.4、收尾：终止训练进程，当前task设为3终止，新建一个同dataset_dir的task，重新训练。

    :param task:
    :param precision:
    :param labels_to_names:
    :param train_dir:
    :return:
    """
    print('begin cluster:{}'.format(task.pk))

    # 3.3.1、计算单样本聚类打分，算法：最近3次checkpoint的score，按60%，30%，10%，加权平均。（TODO：这部分可以根据map和category_ap的数据自学习）
    print('3.3.1')
    use_steps = []
    db_steps = ClusterEvalStep.objects.filter(train_task_id=task.pk).order_by('-checkpoint_step')
    for step in db_steps:
        if step.checkpoint_step in use_steps:
            continue
        use_steps.append(step.checkpoint_step)
        if len(use_steps) == 3:
            break

    print('use_steps:')
    print(use_steps)

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

    print('sample_scores:')
    print(sample_scores)

    ClusterSampleScore.objects.filter(train_task_id=task.pk).delete()
    ClusterUpcScore.objects.filter(train_task_id=task.pk).delete()
    for sample_serial in sample_scores:
        one_sample = sample_scores[sample_serial]
        detections = {}
        groundtruth_label = -1
        for checkpoint_step in one_sample:
            groundtruth_label = one_sample[checkpoint_step]['groundtruth_label']
            weight = 0.0
            if checkpoint_step == use_steps[0]:
                weight = .5
            elif checkpoint_step == use_steps[1]:
                weight = .3
            elif checkpoint_step == use_steps[2]:
                weight = .2

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
                upc_1=labels_to_names[groundtruth_label],
                upc_2=labels_to_names[detection],
                score=detections[detection]
            )

    # 3.3.2、聚类打分算法：取A->B或B->A单样本最高值作为聚类结果
    print('3.3.2')
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
                if upc_1 == upc_2:
                    raise ValueError('error: cluster with same upc: {}'.format(upc_1))
                ClusterUpcScore.objects.create(
                    train_task_id=task.pk,
                    upc_1=upc_1,
                    upc_2=upc_2,
                    score=upc_scores[upc_1][upc_2]
                )

    # 3.3.3.1、聚类：设定45%为阀值进行聚类连接
    print('3.3.3.1')
    upc_scores = ClusterUpcScore.objects.filter(train_task_id=task.pk)
    source_cluster = {}  # {A:[B,C,D],B:[C,E],C:[F],E:[F],G:[A,H],M:[O,N]}
    for upc_score in upc_scores:
        if upc_score.score < 0.45:
            continue
        if upc_score.upc_1 in source_cluster:
            if upc_score.upc_2 in source_cluster[upc_score.upc_1]:
                continue
            else:
                source_cluster[upc_score.upc_1].append(upc_score.upc_2)
        else:
            source_cluster[upc_score.upc_1] = [upc_score.upc_2]

    print('source_cluster:')
    print(source_cluster)

    sorted_cluster = {}  # {A:[B,C,D,G,H],B:[C,E],C:[F],E:[F],M:[N,O]}
    for key in source_cluster:
        ones = [key]
        for one in source_cluster[key]:
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

    print('sorted_cluster:')
    print(sorted_cluster)

    def inner_find(f_upc, sorted_cluster, solved_keys):
        solved_keys.append(f_upc)
        ones = sorted_cluster[f_upc]
        ret = ones.copy()
        for one in ones:
            if one in sorted_cluster:
                finds = inner_find(one, sorted_cluster, solved_keys)
                for find_one in finds:
                    if find_one not in ret:
                        ret.append(find_one)

        return ret

    solved_cluster = {}  # {A:[B,C,D,E,F,G,H],M:[N,O]}
    solved_keys = []
    for f_upc in sorted_cluster:
        solved_cluster[f_upc] = inner_find(f_upc, sorted_cluster, solved_keys)

    print('solved_cluster:')
    print(solved_cluster)

    # 3.3.3.2、判断是否聚类后分类数限制，然后退出
    # 分类小于8，需要处理为继续训练。继续训练需要删除本次训练的cluster_sample_score和cluster_upc_score
    print('3.3.3.2')
    cur_category_cnt = task.category_cnt
    for f_upc in solved_cluster:
        cur_category_cnt = cur_category_cnt - len(solved_cluster[f_upc])

    print('category_cnt:{}/{}'.format(cur_category_cnt, task.category_cnt))
    if cur_category_cnt < 8 and cur_category_cnt/task.category_cnt < .05:
        print('exit for cur_category_cnt:{}/{}'.format(cur_category_cnt, task.category_cnt))
        task.restart_cnt = task.restart_cnt+1
        task.m_ap = precision
        task.save()
        return

    # 3.3.3.3、修改目录存储结构 重启问题，目录结构调整方案分为拷贝目录和拷贝子目录
    print('3.3.3.3')
    def move_file(f_upc,src_dir,dest_dir):
        f_structure = ClusterStructure.objects.filter(f_upc=f_upc).order_by('id')[:1]
        if len(f_structure)>0:
            for one_move_dir in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir,one_move_dir), dest_dir)
        else:
            shutil.move(src_dir, dest_dir)

    tmp_dir = os.path.join(task.dataset_dir, 'tmp')
    if not tf.gfile.Exists(tmp_dir):
        tf.gfile.MakeDirs(tmp_dir)
    for father in solved_cluster:
        father_dir = os.path.join(task.dataset_dir,father)
        father_tmp_dir = os.path.join(tmp_dir,father)
        shutil.move(father_dir,tmp_dir)
        tf.gfile.MakeDirs(father_dir)
        move_file(father, father_tmp_dir, father_dir)
        for upc in solved_cluster[father]:
            upc_dir = os.path.join(task.dataset_dir,upc)
            move_file(upc, upc_dir, father_dir)
    os.rmdir(tmp_dir)

    # 3.3.3.4、记录到cluster_structure表中，重启问题，原cluster会进一步收拢，
    # 每个结构在创建的时候已经建设
    print('3.3.3.4')
    for father in solved_cluster:
        for upc in solved_cluster[father]:
            # 上次训练已经有聚类 TODO 训练必须从上往下训练
            last_f_structures = ClusterStructure.objects.filter(f_upc=upc).order_by('id')
            if len(last_f_structures)>0:
                for last_child_structure in last_f_structures:
                    last_child_structure.f_upc = father
                    last_child_structure.update_train_task_id = task.pk
                    last_child_structure.save()

            # 修改自己
            c_structure = ClusterStructure.objects.filter(upc=upc).order_by('id')[:1]
            c_structure[0].f_upc = father
            c_structure[0].update_train_task_id = task.pk
            c_structure[0].save()

    # 3.3.4、收尾：终止训练进程，当前task设为3终止，新建一个同dataset_dir的task，重新训练。
    print('3.3.4')
    train_ps = os.popen('ps -ef | grep train.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if train_ps != '':
        pid = int(train_ps.split()[1])
        os.system('kill -s 9 {}'.format(str(pid)))

    task.state=3
    task.m_ap=precision
    task.save()

    TrainTask.objects.create(
        dataset_dir=task.dataset_dir,
    )

    print('Finished cluster: {}'.format(task.dataset_dir))

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
            # 训练完成，创建子训练任务
            for f_upc in os.listdir(task.dataset_dir):
                father_dir = os.path.join(task.dataset_dir,f_upc)
                if os.path.isdir(father_dir):
                    children = ClusterStructure.objects.filter(f_upc=f_upc)
                    if len(children) >= 2:
                        TrainTask.objects.create(
                            dataset_dir=father_dir,
                        )
