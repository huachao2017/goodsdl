import logging
import os
import shutil
from django.conf import settings
import tensorflow as tf
from .models import Image, ImageResult, ImageGroundTruth, TrainImage, TrainUpc, TaskLog, TrainAction, TrainActionUpcs, \
    TrainModel
from . import common
import socket
from . import convert_goods
import datetime
import subprocess

logger = logging.getLogger('cron')


def test():
    logger.info('test cron')


def transfer_sample():
    doing_qs = TaskLog.objects.filter(name='transfer_sample').filter(state=1)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='transfer_sample',
    )

    try:
        logger.info('transfer_sample: begin task')
        ret = _do_transfer_sample()
        logger.info('transfer_sample: end task')
    except Exception as e:
        cur_task.state = 20
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = 10
        cur_task.message = ret
        cur_task.save()


def _do_transfer_sample():
    # 查找需要转化的来自前端检测的Image
    train_image_qs = TrainImage.objects.filter(source_image_id__gt=0).order_by('-id')
    if len(train_image_qs) == 0:
        image_qs = Image.objects.all()
    else:
        last_train_image = train_image_qs[0]
        last_image = Image.objects.get(id=last_train_image.source_image.pk)
        image_qs = Image.objects.filter(id__gt=last_image.pk).exclude(
            image_ground_truth_id=last_image.image_group_truth.pk)

    # 将Image列表转化为dict: key=identify，value=Image[]
    identify_to_images = {}
    for image in image_qs:
        # 去除没有确定upc的
        if image.image_ground_truth is not None:
            if image.identify in identify_to_images:
                identify_to_images[image.identify].append(image)
            else:
                identify_to_images[image.identify] = [image, ]

    total_example_cnt = 0
    for identify in identify_to_images:
        # 处理同一个identify的Image[]
        false_example = False
        true_max_score = 0
        true_image = None
        image_ground_truth = None
        example_cnt = 0
        for image in identify_to_images[identify]:
            image_ground_truth = image.image_ground_truth
            image_result_qs = image.image_results.filter(upc=image_ground_truth.groundtruth_upc)
            if len(image_result_qs) == 0:
                # false example 只加一个
                if not false_example:
                    train_source = '{}/{}/{}'.format(common.DATASET_DIR, image_ground_truth.groundtruth_upc,
                                                     os.path.basename(image.source.path))
                    shutil.copy(image.source.path, os.path.join(settings.MEDIA_ROOT, train_source))
                    TrainImage.objects.create(
                        source=train_source,
                        upc=image_ground_truth.groundtruth_upc,
                        source_image_id=image.pk,
                        source_from=2,
                    )
                    example_cnt += 1
                    total_example_cnt += 1
                    false_example = True
                    logger.info('transfer_sample: add one false example')
            else:
                # true example 加score最高的一个
                image_result = image_result_qs[0]
                if image_result.score > true_max_score:
                    true_max_score = image_result.score
                    true_image = image

        if true_image is not None:
            train_source = '{}/{}/{}'.format(common.DATASET_DIR, image_ground_truth.groundtruth_upc,
                                             os.path.basename(true_image.source.path))
            shutil.copy(true_image.source.path, os.path.join(settings.MEDIA_ROOT, train_source))
            TrainImage.objects.create(
                source=train_source,
                upc=image_ground_truth.groundtruth_upc,
                source_image_id=true_image.pk,
                source_from=2,
            )
            example_cnt += 1
            total_example_cnt += 1
            logger.info('transfer_sample: add one true example')

        if false_example or true_image is not None:
            # add or update TrainUpc
            try:
                train_upc = TrainUpc.objects.get(upc=image_ground_truth.groundtruth_upc)
                train_upc.cnt += example_cnt
                train_upc.save()
            except TrainUpc.DoesNotExist as e:
                TrainUpc.objects.create(
                    upc=image_ground_truth.groundtruth_upc,
                    cnt=example_cnt,
                )

    return '成功转化{}个样本'.format(total_example_cnt)


def create_train():
    doing_qs = TaskLog.objects.filter(name='create_train').filter(state=1)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='create_train',
    )

    try:
        logger.info('create_train: begin task')
        ret = _do_create_train()
        logger.info('create_train: end task')
    except Exception as e:
        cur_task.state = 20
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = 10
        cur_task.message = ret
        cur_task.save()


def _do_create_train():
    _do_create_train_ta()
    # TF & TC
    last_t_qs = TrainAction.objects.filter(state__gte=10).order_by('-id')
    if len(last_t_qs) > 0:
        last_t = last_t_qs[0]
        train_model_qs = TrainModel.objects.filter(train_action_id=last_t.pk).order_by('-id')
        f_train_model = train_model_qs[0]
        doing_tf_qs = TrainAction.objects.filter(action='TF').filter(state__lt=10).order_by('-id')
        doing_tc_qs = TrainAction.objects.filter(action='TC').filter(state__lt=10).order_by('-id')
        f_train_upcs = last_t.upcs.all()
        train_upcs = TrainUpc.objects.all()

        upcs = []
        for train_upc in train_upcs:
            upcs.append(train_upc.upc)

        upcs = sorted(upcs)

        f_upcs = []
        for train_upc in f_train_upcs:
            f_upcs.append(train_upc.upc)

        # 根据train_upcs和f_train_upcs进行样本筛选
        append_upcs = []
        for upc in upcs:
            if upc not in f_upcs:
                append_upcs.append(upc)

        train_image_qs = TrainImage.objects.filter(create_time__gt=last_t.create_time)
        if len(append_upcs) > 0:
            if len(doing_tc_qs) == 0 and len(train_image_qs) > 10:
                if len(doing_tf_qs) > 0:
                    # 退出正在训练的TF
                    doing_tf = doing_tf_qs[0]
                    doing_tf.state = 9
                    doing_tf.save()
                logger.info('create_train: TC,新增类（{}）,新增样本（{}）'.format(len(append_upcs), len(train_image_qs)))
                _create_train('TC', f_train_model.pk)
        else:
            if len(doing_tf_qs) == 0:
                now = datetime.datetime.now()
                if (now - last_t.create_time).days >= 1 or len(train_image_qs) > 200:
                    logger.info('create_train: TF,新增样本（{}）,间距天数（{}）'.format(len(train_image_qs),
                                                                            (now - last_t.create_time).days))
                    _create_train('TF', f_train_model.pk)

    return ''


def _do_create_train_ta():
    # TA
    doing_ta = TrainAction.objects.filter(action='TA').filter(state__lt=10)
    if len(doing_ta) == 0:
        last_ta = None
        last_ta_qs = TrainAction.objects.filter(action='TA').filter(state__gte=10).order_by('-id')
        if len(last_ta_qs) > 0:
            last_ta = last_ta_qs[0]

        if last_ta is None:
            # 样本有2000个
            train_image_qs = TrainImage.objects.all()
            if len(train_image_qs) >= 2000:
                logger.info('create_train: TA,新增样本（{}）'.format(len(train_image_qs)))
                _create_train('TA', 0)
        else:
            # 间距达到7天或者新增样本超过2000个
            now = datetime.datetime.now()
            train_image_qs = TrainImage.objects.filter(create_time__gt=last_ta.create_time)
            if (now - last_ta.create_time).days >= 7 or len(train_image_qs) >= 2000:
                logger.info('create_train: TA,新增样本（{}）,间距天数（{}）'.format(len(train_image_qs),
                                                                        (now - last_ta.create_time).days))
                _create_train('TA', 0)


def _create_train(action, f_model_id):
    train_action = TrainAction.objects.create(
        action=action,
        f_model_id=f_model_id,
        desc=''
    )

    train_action.train_path = os.path.join(common.TRAIN_DIR, str(train_action.pk))
    # 数据准备
    names_to_labels, training_filenames, validation_filenames = convert_goods.prepare_train(train_action)
    # 更新数据
    # 'upcs'
    for upc in names_to_labels:
        train_upc = TrainUpc.objects.get(upc=upc)
        TrainActionUpcs.objects.create(
            train_action_id=train_action.pk,
            train_upc=train_upc,
            upc=upc,
            cnt=train_upc.cnt,
        )
    train_action.train_cnt = len(training_filenames)
    train_action.validation_cnt = len(validation_filenames)
    # 'devcice'
    if train_action.action == 'TC':
        pass

    train_action.save()


def execute_train():
    doing_qs = TaskLog.objects.filter(name='execute_train').filter(state=1)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='execute_train',
    )

    try:
        logger.info('execute_train: begin task')
        ret = _do_execute_train()
        logger.info('execute_train: end task')
    except Exception as e:
        cur_task.state = 20
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = 10
        cur_task.message = ret
        cur_task.save()


def _do_execute_train():
    my_ip = get_host_ip()
    if my_ip == '192.168.1.170':
        quit_train_qs = TrainAction.objects.filter(state=9).exclude(action='TC')
        for quit_train in quit_train_qs:
            stop_train(quit_train)
            quit_train.state = 20
            quit_train.save()

        begin_train_qs = TrainAction.objects.filter(state=1).exclude(action='TC')
        for begin_train in begin_train_qs:
            begin_train(begin_train)
            begin_train.state = 5
            begin_train.save()
    elif my_ip == '192.168.1.173':
        quit_train_qs = TrainAction.objects.filter(state=9).filter(action='TC')
        for quit_train in quit_train_qs:
            stop_train(quit_train)
            quit_train.state = 20
            quit_train.save()
        begin_train_qs = TrainAction.objects.filter(state=1).filter(action='TC')
        for begin_train in begin_train_qs:
            begin_train(begin_train)
            begin_train.state = 5
            begin_train.save()

    return ''


def begin_train(train_action):
    train_cuda_visible_devices = ''
    if train_action.action == 'TA':
        train_cuda_visible_devices = '0'
    elif train_action.action == 'TF':
        train_cuda_visible_devices = '1'
    elif train_action.action == 'TC':
        train_cuda_visible_devices = '0'

    # 训练
    if train_action.action == 'TC':
        checkpoint_path = os.path.join(common.MODEL_DIR, str(train_action.f_model.pk))
        command = 'nohup python3 {}/step2/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  '--checkpoint_path={} --checkpoint_exclude_scopes=final_layer,aux_11/aux_logits/FC --trainable_scopes=final_layer,aux_11/aux_logits/FC' \
                  ' > /root/train_{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_action.train_path,
            train_action.train_path,
            train_action.train_cnt,
            'nasnet_large',
            1,
            8,
            train_cuda_visible_devices,
            checkpoint_path,
            train_action.action
        )
    elif train_action.action == 'TF':
        checkpoint_path = os.path.join(common.MODEL_DIR, str(train_action.f_model.pk))
        command = 'nohup python3 {}/step2/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  '--checkpoint_path={} > /root/train_{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_action.train_path,
            train_action.train_path,
            train_action.train_cnt,
            'nasnet_large',
            1,
            8,
            train_cuda_visible_devices,
            checkpoint_path,
            train_action.action
        )
    else:
        command = 'nohup python3 {}/step2/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  ' > /root/train_{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_action.train_path,
            train_action.train_path,
            train_action.train_cnt,
            'nasnet_large',
            1,
            8,
            train_cuda_visible_devices,
            train_action.action
        )
    logger.info(command)
    subprocess.call(command, shell=True)
    # 评估
    command = 'nohup python3 {}/step2/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={}' \
              ' > /root/eval_{}.out 2>&1 &'.format(
        os.path.join(settings.BASE_DIR, 'dl'),
        train_action.train_path,
        os.path.join(
            settings.MEDIA_ROOT,
            common.DATASET_DIR),
        train_action.train_path,
        os.path.join(train_action.train_path, 'eval_log'),
        train_action.validation_cnt,
        'nasnet_large',
        train_action.action,
    )
    logger.info(command)
    subprocess.call(command, shell=True)


def stop_train(train_action):
    train_dir = os.path.join(common.TRAIN_DIR, str(train_action.pk))
    train_ps = os.popen('ps -ef | grep train.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if train_ps != '':
        pid = int(train_ps.split()[1])
        os.system('kill -s 9 {}'.format(str(pid)))
    eval_ps = os.popen('ps -ef | grep eval2.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if eval_ps != '':
        pid = int(eval_ps.split()[1])
        os.system('kill -s 9 {}'.format(str(pid)))


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip
