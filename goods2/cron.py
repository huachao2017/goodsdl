import logging
import os
import shutil
from django.db.models import Max, Count
from django.conf import settings
import tensorflow as tf
from goods2.models import Deviceid, DeviceidPrecision, DeviceidTrain, Image, ImageResult, ImageGroundTruth, TrainImage, TaskLog, TrainAction, TrainActionUpcs, \
    TrainModel, EvalLog
from . import common
import socket
from goods2 import convert_goods
import datetime
import subprocess
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

logger = logging.getLogger('cron')


def test():
    logger.info('test cron')


def check_device():
    doing_qs = TaskLog.objects.filter(name='check_device').filter(state=common.TRAIN_STATE_WAITING)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='check_device',
        ip=get_host_ip(),
    )

    try:
        logger.debug('check_device: begin task')
        ret = _do_check_device()
        logger.debug('check_device: end task')
    except Exception as e:
        logger.error('check_device: {}'.format(e))
        logger.error(traceback.format_exc())
        cur_task.state = common.TASK_STATE_ERROR
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = common.TASK_STATE_COMPLETE
        cur_task.message = ret
        cur_task.save()


def _do_check_device():
    device_qs = Deviceid.objects.all()
    for device in device_qs:
        device_precision_qs = device.device_precisions.order_by('-id')
        if len(device_precision_qs) == 0:
            image_ground_truth_qs = ImageGroundTruth.objects.filter(deviceid=device.deviceid).order_by('-id')
            if len(image_ground_truth_qs)>0:
                _do_check_one_device(device, image_ground_truth_qs)
        else:
            last_device_precision = device_precision_qs[0]
            image_ground_truth_qs = ImageGroundTruth.objects.filter(deviceid=device.deviceid).filter(create_time__gt=last_device_precision.create_time).order_by('-id')
            if len(image_ground_truth_qs) > 0:
                _do_check_one_device(device, image_ground_truth_qs)


def _do_check_one_device(device, image_ground_truth_qs):
    total_truth_rate = 0.0
    total_precision = 0.0
    for image_group_truth in image_ground_truth_qs:
        total_truth_rate += image_group_truth.truth_rate
        total_precision += image_group_truth.precision

    precision = total_precision /len(image_ground_truth_qs)
    truth_rate = total_truth_rate / len(image_ground_truth_qs)
    if device.state < common.DEVICE_STATE_COMMERCIAL and truth_rate > 0.95:
        device.state = common.DEVICE_STATE_COMMERCIAL
        device.commercial_time = datetime.datetime.now()
        device.save()
    DeviceidPrecision.objects.create(
        device_id=device.pk,
        truth_rate=truth_rate,
        precision=precision
    )


def transfer_sample():
    doing_qs = TaskLog.objects.filter(name='transfer_sample').filter(state=common.TASK_STATE_DOING)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='transfer_sample',
        ip=get_host_ip(),
    )

    try:
        logger.debug('transfer_sample: begin task')
        ret = _do_transfer_sample()
        logger.debug('transfer_sample: end task')
    except Exception as e:
        logger.error('transfer_sample: {}'.format(e))
        logger.error(traceback.format_exc())
        cur_task.state = common.TASK_STATE_ERROR
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = common.TASK_STATE_COMPLETE
        cur_task.message = ret
        cur_task.save()


def _do_transfer_sample():
    # 查找需要转化的来自前端检测的Image
    train_image_max_qs = TrainImage.objects.filter(source_image_id__gt=0).values_list('deviceid').annotate(ct=Max('create_time')).order_by('-ct')
    train_image_group_list = list(zip(*train_image_max_qs))
    image_group_qs = Image.objects.filter(image_ground_truth_id__gt=0).values_list('deviceid').annotate(cnt=Count('id')).order_by('cnt')

    logger.info('transfer image device count: {}'.format(len(image_group_qs)))

    total_example_cnt = 0
    for image_group in image_group_qs:
        deviceid = image_group[0]
        if len(train_image_max_qs)>0 and deviceid in train_image_group_list[0]:
            index = train_image_group_list[0].index(deviceid)
            image_qs = Image.objects.filter(deviceid=deviceid).filter(image_ground_truth_id__gt=0).filter(create_time__gt=train_image_group_list[1][index])
        else:
            image_qs = Image.objects.filter(deviceid=deviceid).filter(image_ground_truth_id__gt=0)
        # 将Image列表转化为dict: key=identify，value=Image[]
        identify_to_images = {}
        for image in image_qs:
            if image.identify in identify_to_images:
                identify_to_images[image.identify].append(image)
            else:
                identify_to_images[image.identify] = [image, ]

        for identify in identify_to_images:
            # 处理同一个identify的Image[]
            false_example = False
            true_max_score = 0
            true_image = None
            image_ground_truth = None
            example_cnt = 0
            for image in identify_to_images[identify]:
                image_ground_truth = image.image_ground_truth
                image_result_qs = image.image_results.filter(upc=image_ground_truth.upc)
                if len(image_result_qs) == 0:
                    # false example 只加一个
                    if not false_example:
                        train_source = '{}/{}/{}/{}'.format(common.get_dataset_dir(), deviceid, image_ground_truth.upc,
                                                         'image_'+os.path.basename(image.source.path))
                        train_source_dir = '{}/{}/{}'.format(common.get_dataset_dir(True), deviceid,
                                                             image_ground_truth.upc)
                        if not tf.gfile.Exists(train_source_dir):
                            tf.gfile.MakeDirs(train_source_dir)
                        train_source_path = '{}/{}'.format(train_source_dir, 'image_'+os.path.basename(image.source.path))
                        try:
                            shutil.copy(image.source.path, train_source_path)
                        except:
                            continue
                        train_image = TrainImage.objects.create(
                            deviceid=deviceid,
                            source=train_source,
                            upc=image_ground_truth.upc,
                            source_image_id=image.pk,
                            source_from=2,
                            score=0.0,
                        )
                        train_image.create_time = image_ground_truth.create_time
                        train_image.save()
                        example_cnt += 1
                        total_example_cnt += 1
                        false_example = True
                        logger.info('[{}]transfer_sample: add one false example'.format(deviceid))
                else:
                    # true example 加score最高的一个
                    image_result = image_result_qs[0]
                    if image_result.score > true_max_score:
                        true_max_score = image_result.score
                        true_image = image

            if true_image is not None:
                train_source = '{}/{}/{}/{}'.format(common.get_dataset_dir(), deviceid, image_ground_truth.upc,
                                                    'image_' +os.path.basename(true_image.source.path))
                train_source_dir = '{}/{}/{}'.format(common.get_dataset_dir(True), deviceid,
                                                         image_ground_truth.upc)
                if not tf.gfile.Exists(train_source_dir):
                    tf.gfile.MakeDirs(train_source_dir)
                train_source_path = '{}/{}'.format(train_source_dir,'image_'+os.path.basename(true_image.source.path))
                try:
                    shutil.copy(true_image.source.path, train_source_path)
                except:
                    continue
                TrainImage.objects.create(
                    deviceid=deviceid,
                    source=train_source,
                    upc=image_ground_truth.upc,
                    source_image_id=true_image.pk,
                    source_from=2,
                    score=true_max_score,
                )
                example_cnt += 1
                total_example_cnt += 1
                logger.info('[{}]transfer_sample: add one true example'.format(deviceid))

    logger.info('成功转化{}个样本'.format(total_example_cnt))
    return '成功转化{}个样本'.format(total_example_cnt)


def create_train():
    doing_qs = TaskLog.objects.filter(name='create_train').filter(state=common.TASK_STATE_DOING)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='create_train',
        ip=get_host_ip()
    )

    try:
        logger.debug('create_train: begin task')
        ret = _do_create_train()
        logger.debug('create_train: end task')
    except Exception as e:
        logger.error('create_train: {}'.format(e))
        logger.error(traceback.format_exc())
        cur_task.state = common.TASK_STATE_ERROR
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = common.TASK_STATE_COMPLETE
        cur_task.message = ret
        cur_task.save()


def _do_create_train():
    doing_ta_tf = TrainAction.objects.exclude(action='TC').filter(state__lte=common.TRAIN_STATE_TRAINING)
    if len(doing_ta_tf) == 0:
        _do_create_train_ta()
    doing_ta_tf = TrainAction.objects.exclude(action='TC').filter(state__lte=common.TRAIN_STATE_TRAINING)
    # TF & TC
    doing_tc = TrainAction.objects.filter(action='TC').filter(state__lte=common.TRAIN_STATE_TRAINING)
    last_t_group_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).values_list('deviceid').annotate(ct=Max('complete_time')).order_by('ct')
    for last_t_group in last_t_group_qs:
        deviceid = last_t_group[0]
        last_t = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).filter(deviceid=deviceid).order_by('-complete_time')[0]

        train_model_qs = TrainModel.objects.filter(train_action_id=last_t.pk).exclude(model_path='').order_by('-id')
        f_train_model = train_model_qs[0]
        f_train_upcs = last_t.upcs.all()
        train_upc_group_qs = TrainImage.objects.filter(deviceid=deviceid).values_list('upc').annotate(cnt=Count('id'))
        upcs = []
        for train_upc_group in train_upc_group_qs:
            upcs.append(train_upc_group[0])
        upcs = sorted(upcs)

        f_upcs = []
        for train_upc in f_train_upcs:
            f_upcs.append(train_upc.upc)

        # 根据train_upcs和f_train_upcs进行样本筛选
        append_upcs = []
        for upc in upcs:
            if upc not in f_upcs:
                append_upcs.append(upc)

        # 只计算注册训练的设备
        train_image_qs = TrainImage.objects.filter(deviceid=deviceid).filter(create_time__gt=last_t.create_time)
        if len(doing_tc)==0 and len(append_upcs) > 0:
            if len(train_image_qs) >= 20:
                pass
                # FIXME 暂时不开启TC
                # if len(doing_tf_qs) > 0:
                    # 退出正在训练的TF
                    # doing_tf = doing_tf_qs[0]
                    # doing_tf.state = 9
                    # doing_tf.save()
                # logger.info('[{}]create_train: TC,新增类（{}）,新增样本（{}）'.format(deviceid, len(append_upcs), len(train_image_qs)))
                # do_create_train('TC', deviceid, f_train_model.pk)
        elif len(doing_ta_tf) == 0 and len(append_upcs) == 0:
            now = datetime.datetime.now()
            # if (now - last_t.complete_time).days >= 1 or len(train_image_qs) >= 50:
            if len(train_image_qs) >= 100:
                logger.info('[{}]create_train: TF,新增样本（{}）,间距天数（{}）'.format(deviceid, len(train_image_qs),
                                                                    (now - last_t.complete_time).days))
                do_create_train('TF', deviceid, f_train_model.pk)

    return ''


def _do_create_train_ta():
    last_ta_group_qs = TrainAction.objects.filter(action='TA').filter(state=common.TRAIN_STATE_COMPLETE).values_list('deviceid').annotate(Max('create_time'))
    # 只计算注册训练的设备
    deviceid_train_qs = DeviceidTrain.objects.all().values('deviceid')
    train_image_group_qs = TrainImage.objects.filter(deviceid__in=deviceid_train_qs).values_list('deviceid').annotate(cnt=Count('id')).order_by('-cnt')

    logger.info('create TA train device count: {}'.format(len(train_image_group_qs)))
    # 新增样本有200个
    last_ta_group_list = list(zip(*last_ta_group_qs))
    now = datetime.datetime.now()
    for train_image_group in train_image_group_qs:
        deviceid = train_image_group[0]
        if len(last_ta_group_qs)>0 and deviceid in last_ta_group_list[0]:
            index = last_ta_group_list[0].index(deviceid)
            last_time = last_ta_group_list[1][index]
            train_image_qs = TrainImage.objects.filter(deviceid=deviceid).filter(create_time__gt=last_time)
            # if (now - last_time).days >= 7 or len(train_image_qs) >= 200:
            if len(train_image_qs) >= 200 or (len(train_image_qs) >= 100 and deviceid == '200'):
                logger.info('[{}]create_train: TA,新增样本（{}）'.format(deviceid, len(train_image_qs)))
                do_create_train('TA', deviceid, None)
                return
        elif train_image_group[1] >= 200:
            logger.info('[{}]create_train: TA,新增样本（{}）'.format(deviceid, train_image_group[1]))
            do_create_train('TA', deviceid, None)
            return


def do_create_train(action, deviceid, f_model_id):
    train_action = TrainAction.objects.create(
        action=action,
        deviceid=deviceid,
        f_model_id=f_model_id,
        desc=''
    )

    train_action.train_path = os.path.join(common.get_train_path(), str(train_action.pk))
    # 数据准备
    names_to_labels, training_filenames, validation_filenames = convert_goods.prepare_train(train_action)
    train_action = TrainAction.objects.get(pk=train_action.pk)
    train_action.train_path = os.path.join(common.get_train_path(), str(train_action.pk))

    if names_to_labels is None:
        train_action.state = common.TRAIN_STATE_COMPLETE_WITH_ERROR
        logger.info('update_train_after_error_create')
        train_action.save()
        return

    train_image_group_qs = TrainImage.objects.filter(deviceid=deviceid).values_list('upc').annotate(
        cnt=Count('id')).order_by('-cnt')
    train_image_group_list = list(zip(*train_image_group_qs))
    # 更新数据
    # 'upcs'
    for upc in names_to_labels:
        try:
            index = train_image_group_list[0].index(upc)
            TrainActionUpcs.objects.create(
                train_action_id=train_action.pk,
                upc=upc,
                cnt=train_image_group_list[1][index],
            )
        except:
            logger.warning('can not find upc:{}'.format(upc))
            continue
    train_action.train_cnt = len(training_filenames)
    train_action.validation_cnt = len(validation_filenames)
    # 'device'
    # if train_action.action == 'TC':
    #     pass

    logger.info('update_train_after_create')
    train_action.save()
    return train_action


def do_test_train(action):
    deviceid = '3540'
    train_action = TrainAction.objects.create(
        action=action,
        deviceid=deviceid,
        desc=''
    )

    train_action.train_path = os.path.join(common.get_train_path(), str(train_action.pk))
    # 数据准备
    names_to_labels, training_filenames, validation_filenames = convert_goods.prepare_test_train(train_action)

    if names_to_labels is None:
        train_action.state = common.TRAIN_STATE_COMPLETE_WITH_ERROR
        train_action.save()
        return

    train_action.train_cnt = len(training_filenames)
    train_action.validation_cnt = len(validation_filenames)
    train_action.deviceid = '103540' # 修订为测试训练deviceid, 不干扰原训练样本
    train_action.save()
    return train_action


def do_create_train_bind(action, deviceid, f_model_id, bind_deviceid_list):
    train_action = TrainAction.objects.create(
        action=action,
        deviceid=deviceid,
        f_model_id=f_model_id,
        desc=''
    )

    train_action.train_path = os.path.join(common.get_train_path(), str(train_action.pk))
    # 数据准备
    names_to_labels, training_filenames, validation_filenames = convert_goods.prepare_train(train_action, bind_deviceid_list=bind_deviceid_list)
    train_action = TrainAction.objects.get(pk=train_action.pk)
    train_action.train_path = os.path.join(common.get_train_path(), str(train_action.pk))

    if names_to_labels is None:
        train_action.state = common.TRAIN_STATE_COMPLETE_WITH_ERROR
        logger.info('update_train_after_error_create')
        train_action.save()
        return

    train_image_group_qs = TrainImage.objects.filter(deviceid__in=bind_deviceid_list).values_list('upc').annotate(
        cnt=Count('id')).order_by('-cnt')
    train_image_group_list = list(zip(*train_image_group_qs))
    # 更新数据
    # 'upcs'
    for upc in names_to_labels:
        try:
            index = train_image_group_list[0].index(upc)
            TrainActionUpcs.objects.create(
                train_action_id=train_action.pk,
                upc=upc,
                cnt=train_image_group_list[1][index],
            )
        except:
            logger.warning('can not find upc:{}'.format(upc))
            continue
    train_action.train_cnt = len(training_filenames)
    train_action.validation_cnt = len(validation_filenames)
    # 'device'
    # if train_action.action == 'TC':
    #     pass

    logger.info('update_train_after_create')
    train_action.save()
    return train_action


def execute_train():
    doing_qs = TaskLog.objects.filter(name='execute_train').filter(state=common.TASK_STATE_DOING)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='execute_train',
        ip=get_host_ip(),
    )

    try:
        logger.debug('execute_train: begin task')
        ret = _do_execute_train()
        logger.debug('execute_train: end task')
    except Exception as e:
        logger.error('execute_train: {}'.format(e))
        logger.error(traceback.format_exc())
        cur_task.state = common.TASK_STATE_ERROR
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = common.TASK_STATE_COMPLETE
        cur_task.message = ret
        cur_task.save()


def _do_execute_train():
    my_ip = get_host_ip()
    if my_ip == '192.168.1.60':
        quit_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_STOP).exclude(action='TC').filter(ip=my_ip)
        for quit_train in quit_train_qs:
            common.stop_train_ps(quit_train)
            quit_train.state = common.TRAIN_STATE_COMPLETE_WITH_STOP
            quit_train.complete_time = datetime.datetime.now()
            logger.info('update_train_after_quit')
            quit_train.save()

        training_ta_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_TRAINING).filter(action='TA').filter(ip=my_ip).order_by('id')
        if len(training_ta_train_qs) <= 0:
            begin_ta_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_WAITING).filter(action='TA').order_by('id')
            for begin_train in begin_ta_train_qs:
                train_command, eval_command = _do_begin_train(begin_train)
                update_begin_train_after_execute(begin_train, train_command, eval_command, my_ip)
                # only can do one
                break

        training_tf_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_TRAINING).filter(action='TF').filter(ip=my_ip).order_by('id')
        if len(training_tf_train_qs) <= 0:
            begin_tf_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_WAITING).filter(
                action='TF').order_by('id')
            for begin_train in begin_tf_train_qs:
                train_command, eval_command = _do_begin_train(begin_train)
                update_begin_train_after_execute(begin_train, train_command, eval_command, my_ip)
                # only can do one
                break
    elif my_ip == '192.168.1.73':
        quit_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_STOP).filter(action='TC').filter(ip=my_ip)
        for quit_train in quit_train_qs:
            common.stop_train_ps(quit_train)
            quit_train.state = common.TRAIN_STATE_COMPLETE_WITH_STOP
            quit_train.complete_time = datetime.datetime.now()
            logger.info('update_train_after_quit')
            quit_train.save()
        training_tc_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_TRAINING).filter(action='TC').filter(ip=my_ip).order_by('id')
        if len(training_tc_train_qs) <= 0:
            begin_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_WAITING).filter(action='TC').order_by('id')
            for begin_train in begin_train_qs:
                train_command, eval_command = _do_begin_train(begin_train)
                update_begin_train_after_execute(begin_train, train_command, eval_command, my_ip)
                # only can do one
                break

    return ''


def update_begin_train_after_execute(begin_train, train_command, eval_command, my_ip):
    if train_command is not None:
        begin_train.ip = my_ip
        begin_train.train_command = train_command
        begin_train.eval_command = eval_command
        begin_train.state = common.TRAIN_STATE_TRAINING
        logger.info('update_train_after_execute')
        begin_train.save()


def _do_begin_train(train_action):

    # 防止样本生成时间过长没有完成
    if train_action.train_path is None or train_action.train_path == '':
        return None,None

    # 训练
    if train_action.action == 'TC':
        f_model_id = train_action.f_model.pk
        checkpoint_path = os.path.join(common.get_model_path(), str(f_model_id))
        train_command = 'nohup python3 {}/goods2/dl/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  '--checkpoint_path={} --checkpoint_exclude_scopes=final_layer,aux_11/aux_logits/FC --trainable_scopes=final_layer,aux_11/aux_logits/FC' \
                  ' > /root/train_{}.out 2>&1 &'.format(
            settings.BASE_DIR,
            train_action.train_path,
            train_action.train_path,
            train_action.train_cnt,
            'nasnet_large',
            1,
            8,
            '0',
            checkpoint_path,
            train_action.action
        )
    elif train_action.action == 'TF':
        checkpoint_path = os.path.join(common.get_model_path(), str(train_action.f_model.pk))
        train_command = 'nohup python3 {}/goods2/dl/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  '--learning_rate=0.003 --checkpoint_path={} > /root/train_{}.out 2>&1 &'.format(
            settings.BASE_DIR,
            train_action.train_path,
            train_action.train_path,
            train_action.train_cnt,
            'nasnet_large',
            2,
            8,
            '0,1',
            checkpoint_path,
            train_action.action
        )
    else:
        # FIXME 暂时不开启TF和TC
        train_command = 'nohup python3 {}/goods2/dl/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  ' > /root/train_{}.out 2>&1 &'.format(
            settings.BASE_DIR,
            train_action.train_path,
            train_action.train_path,
            train_action.train_cnt,
            'nasnet_large',
            2,
            8,
            '0,1',
            train_action.action
        )
    subprocess.call(train_command, shell=True)
    # 评估
    eval_command = 'nohup python3 {}/goods2/dl/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={}' \
              ' > /root/eval_{}.out 2>&1 &'.format(
        settings.BASE_DIR,
        train_action.train_path,
        os.path.join(common.get_dataset_dir(True),train_action.deviceid),
        train_action.train_path,
        os.path.join(train_action.train_path, 'eval_log'),
        train_action.validation_cnt,
        'nasnet_large',
        train_action.action,
    )
    subprocess.call(eval_command, shell=True)
    return train_command, eval_command


def check_train():
    doing_qs = TaskLog.objects.filter(name='check_train').filter(state=common.TASK_STATE_DOING)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='check_train',
        ip=get_host_ip(),
    )

    try:
        logger.debug('check_train: begin task')
        ret = _do_check_train()
        logger.debug('check_train: end task')
    except Exception as e:
        logger.error('check_train: {}'.format(e))
        logger.error(traceback.format_exc())
        cur_task.state = common.TASK_STATE_ERROR
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = common.TASK_STATE_COMPLETE
        cur_task.message = ret
        cur_task.save()

def _do_check_train():
    my_ip = get_host_ip()
    ret = ''
    if my_ip == '192.168.1.60':
        doing_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_TRAINING).filter(ip=my_ip).exclude(action='TC')
        for check_train in doing_train_qs:
            _do_check_one_train(check_train)
    elif my_ip == '192.168.1.73':
        doing_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_TRAINING).filter(ip=my_ip).filter(action='TC')
        for check_train in doing_train_qs:
            _do_check_one_train(check_train)

    return ret


def get_max_precision_interval(eval_log_qs, history_cnt=5):
    history_precisions = np.array((eval_log_qs[1].precision))
    for i in range(history_cnt-1):
        history_precisions = np.append(history_precisions,eval_log_qs[i+2].precision)
    return np.max(np.abs(history_precisions-eval_log_qs[0].precision))


# 生成信号：
# precision达0.9后：
# TA：step超过5000且有5次eval，最后一次和前10次精度上升小于0.5%
# TF：step超过1000且有5次eval，最后一次和前10次精度上升小于1%
# TC：step超过500且有3次eval，最后一次和前5次精度上升小于1%
def _do_check_one_train(train_action):
    train_pid = common.get_train_pid(train_action)
    if train_pid == 0:
        train_action.state=common.TRAIN_STATE_COMPLETE_WITH_ERROR
        logger.info('update_train_after_error_complete')
        train_action.save()
        logger.error('train process has been killed:{};'.format(train_action.pk))
        return 'train process has been killed:{};'.format(train_action.pk)

    _syn_event_log(train_action)
    eval_log_qs = EvalLog.objects.filter(train_action_id=train_action.pk).order_by('-id')

    if len(eval_log_qs)>0 and eval_log_qs[0].precision>=0.95:
        last_eval_log = eval_log_qs[0]
        precision_interval = 1
        check_cnt = 3
        if train_action.action == 'TA':
            if len(eval_log_qs)>check_cnt and eval_log_qs[check_cnt].precision>=0.99:
                precision_interval = get_max_precision_interval(eval_log_qs, history_cnt=check_cnt)
            if last_eval_log.checkpoint_step>=10000 and precision_interval<=0.001:
                _do_create_train_model(train_action, last_eval_log.checkpoint_step,last_eval_log.precision)
        elif train_action.action == 'TF':
            if len(eval_log_qs)>check_cnt and eval_log_qs[check_cnt].precision>=0.98:
                precision_interval = get_max_precision_interval(eval_log_qs)
            if last_eval_log.checkpoint_step>=1000 and precision_interval<=0.005:
                _do_create_train_model(train_action, last_eval_log.checkpoint_step,last_eval_log.precision)
        elif train_action.action == 'TC':
            if len(eval_log_qs)>check_cnt and eval_log_qs[check_cnt].precision>=0.98:
                precision_interval = get_max_precision_interval(eval_log_qs)
            if last_eval_log.checkpoint_step>=500 and precision_interval<=0.01:
                _do_create_train_model(train_action, last_eval_log.checkpoint_step,last_eval_log.precision)


def _syn_event_log(train_action):
    eval_dir = os.path.join(train_action.train_path, 'eval_log')
    if not os.path.isdir(eval_dir):
        logger.error('not found eval log: {}'.format(eval_dir))
        # FIXME need raise error
        return
    event_acc = EventAccumulator(eval_dir)
    event_acc.Reload()
    w_times, step_nums, vals = zip(*event_acc.Scalars('PASCAL/Precision/mAP'))

    eval_log_qs = EvalLog.objects.filter(train_action_id=train_action.pk).order_by('-id')
    last_eval_log = None
    if len(eval_log_qs)>0:
        last_eval_log = eval_log_qs[0]

    for i in range(len(w_times)):
        if last_eval_log is None or last_eval_log.checkpoint_step < step_nums[i]:
            EvalLog.objects.create(
                train_action_id=train_action.pk,
                precision=vals[i],
                checkpoint_step=step_nums[i],
                create_time=datetime.datetime.fromtimestamp(w_times[i])
            )


def _do_create_train_model(train_action, checkpoint_step, precision):
    cur_train_model = TrainModel.objects.create(
        train_action_id=train_action.pk,
        checkpoint_step=checkpoint_step,
        precision=precision,
    )
    _do_export_train(train_action, cur_train_model)
    # if train_action.action == 'TA':
    #     count = 10
    # elif train_action.action == 'TF':
    #     count = 5
    # elif train_action.action == 'TC':
    #     count = 3
    # else:
    #     raise ValueError('Train action type error')
    # train_model_qs = TrainModel.objects.filter(train_action_id=train_action.pk).order_by('-id')[:count]
    # if len(train_model_qs) == count:
    #     min_precision = train_model_qs[0].precision
    #     max_precision = train_model_qs[0].precision
    #     for train_model in train_model_qs:
    #         if min_precision > train_model.precision:
    #             min_precision = train_model.precision
    #         if max_precision < train_model.precision:
    #             max_precision = train_model.precision
    #     if train_action.action == 'TA':
    #         if min_precision >= 0.998 or max_precision - min_precision < 0.002:
    #             _do_export_train(train_action, cur_train_model)
    #     elif train_action.action == 'TF':
    #         if min_precision >= 0.995 or max_precision - min_precision < 0.003:
    #             _do_export_train(train_action, cur_train_model)
    #     elif train_action.action == 'TC':
    #         if min_precision >= 0.99 or max_precision - min_precision < 0.005:
    #             _do_export_train(train_action, cur_train_model)
    return cur_train_model


def _do_export_train(train_action, train_model):
    logger.info('Export <trainid:{}> Graph from classify train.'.format(train_action.pk))
    checkpoint_model_path = tf.train.latest_checkpoint(train_action.train_path)
    if checkpoint_model_path:
        checkpoint_step = checkpoint_model_path.split('-')[-1]
        model_path = os.path.join(common.get_model_path(), str(train_model.pk))
        if not tf.gfile.Exists(model_path):
            tf.gfile.MakeDirs(model_path)
        checkpoint_file_path = os.path.join(model_path, 'checkpoint')
        # 输出pb
        # e2.export(step2_model_name, trained_checkpoint_dir, export_file_path)
        # 重写checkpoint file
        with open(checkpoint_file_path, 'w') as output:
            a = os.path.split(checkpoint_model_path)
            output.write('model_checkpoint_path: "{}"\n'.format(os.path.join(model_path, a[1])))
            output.write('all_model_checkpoint_paths: "{}"\n'.format(os.path.join(model_path, a[1])))
        shutil.copy(checkpoint_model_path + '.data-00000-of-00001', model_path)
        shutil.copy(checkpoint_model_path + '.index', model_path)
        shutil.copy(checkpoint_model_path + '.meta', model_path)
        # copy label
        shutil.copy(os.path.join(train_action.train_path, 'labels.txt'), model_path)

        train_model.checkpoint_step = checkpoint_step
        train_model.model_path = model_path
        train_model.save()
        common.stop_train_ps(train_action)
        train_action.state = common.TRAIN_STATE_COMPLETE
        train_action.complete_time = datetime.datetime.now()
        logger.info('update_train_after_complete')
        train_action.save()
        if train_action.action == 'TA':
            # 'TA'训练完即建立了新的主分支
            old_doing_qs = TrainAction.objects.filter(action__in=['TF', 'TC']).filter(deviceid=train_action.deviceid).filter(state__lte=common.TRAIN_STATE_TRAINING).filter(create_time__lte=train_action.complete_time)
            for old_doing in old_doing_qs:
                old_doing.state = common.TRAIN_STATE_STOP
                old_doing.save()

    else:
        raise ValueError('checkpoint_model_path is None')

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip
