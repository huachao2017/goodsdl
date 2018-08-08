import logging
import os
import shutil
from django.db.models import Max, Count
from django.conf import settings
import tensorflow as tf
from goods2.models import Deviceid, DeviceidPrecision, DeviceidExclude, Image, ImageResult, ImageGroundTruth, TrainImage, TrainUpc, TaskLog, TrainAction, TrainActionUpcs, \
    TrainModel, EvalLog
from . import common
import socket
from goods2 import convert_goods
import datetime
import subprocess
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
        logger.info('check_device: begin task')
        ret = _do_check_device()
        logger.info('check_device: end task')
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
            if len(image_ground_truth_qs) >= 10:
                _do_check_one_device(device, image_ground_truth_qs[:10])
        else:
            last_device_precision = device_precision_qs[0]
            image_ground_truth_qs = ImageGroundTruth.objects.filter(deviceid=device.deviceid).filter(create_time__gt=last_device_precision.create_time).order_by('-id')
            if len(image_ground_truth_qs) >= 10:
                _do_check_one_device(device, image_ground_truth_qs[:10])


def _do_check_one_device(device, image_ground_truth_qs):
    total_truth_rate = 0.0
    total_precision = 0.0
    for image_group_truth in image_ground_truth_qs:
        total_truth_rate += image_group_truth.truth_rate
        total_precision += image_group_truth.precision

    precision = total_precision / 10
    truth_rate = total_truth_rate / 10
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
        logger.info('transfer_sample: begin task')
        ret = _do_transfer_sample()
        logger.info('transfer_sample: end task')
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
    train_image_qs = TrainImage.objects.filter(source_image_id__gt=0).order_by('-id')
    deviceid_exclude_qs = DeviceidExclude.objects.all().values('deviceid')
    if len(train_image_qs) == 0:
        image_qs = Image.objects.exclude(deviceid__in=deviceid_exclude_qs).filter(deviceid=485) # FIXME
    else:
        last_train_image = train_image_qs[0]
        last_image = Image.objects.get(id=last_train_image.source_image.pk)
        image_qs = Image.objects.filter(id__gt=last_image.pk).exclude(deviceid__in=deviceid_exclude_qs).filter(deviceid=485).filter(
            create_time__gt=last_image.create_time)

    logger.info('transfer image length: {}'.format(len(image_qs)))
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
            image_result_qs = image.image_results.filter(upc=image_ground_truth.upc)
            if len(image_result_qs) == 0:
                # false example 只加一个
                if not false_example:
                    train_source = '{}/{}/{}/{}'.format(common.get_dataset_dir(), image_ground_truth.deviceid, image_ground_truth.upc,
                                                     os.path.basename(image.source.path))
                    train_source_dir = '{}/{}/{}'.format(common.get_dataset_dir(True), image_ground_truth.deviceid,
                                                         image_ground_truth.upc)
                    if not tf.gfile.Exists(train_source_dir):
                        tf.gfile.MakeDirs(train_source_dir)
                    train_source_path = '{}/{}'.format(train_source_dir, os.path.basename(image.source.path))
                    try:
                        shutil.copy(image.source.path, train_source_path)
                    except:
                        continue
                    TrainImage.objects.create(
                        deviceid=image_ground_truth.deviceid,
                        source=train_source,
                        upc=image_ground_truth.upc,
                        source_image_id=image.pk,
                        source_from=2,
                        score=0.0,
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
            train_source = '{}/{}/{}/{}'.format(common.get_dataset_dir(), image_ground_truth.deviceid, image_ground_truth.upc,
                                                os.path.basename(true_image.source.path))
            train_source_dir = '{}/{}/{}'.format(common.get_dataset_dir(True), image_ground_truth.deviceid,
                                                     image_ground_truth.upc)
            if not tf.gfile.Exists(train_source_dir):
                tf.gfile.MakeDirs(train_source_dir)
            train_source_path = '{}/{}'.format(train_source_dir,os.path.basename(true_image.source.path))
            try:
                shutil.copy(true_image.source.path, train_source_path)
            except:
                continue
            TrainImage.objects.create(
                deviceid=true_image.deviceid,
                source=train_source,
                upc=image_ground_truth.upc,
                source_image_id=true_image.pk,
                source_from=2,
                score=true_max_score,
            )
            example_cnt += 1
            total_example_cnt += 1
            logger.info('transfer_sample: add one true example')

        if false_example or true_image is not None:
            # add or update TrainUpc
            train_upc_qs = TrainUpc.objects.filter(deviceid=image_ground_truth.deviceid).filter(
                upc=image_ground_truth.upc)
            if len(train_upc_qs) > 0:
                train_upc = train_upc_qs[0]
                train_upc.cnt += 1
                train_upc.save()
            else:
                TrainUpc.objects.create(
                    upc=image_ground_truth.upc,
                    deviceid=image_ground_truth.deviceid,
                    cnt=1,
                )

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
        logger.info('create_train: begin task')
        ret = _do_create_train()
        logger.info('create_train: end task')
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
    _do_create_train_ta()
    # TF & TC
    # last_t_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).order_by('-complete_time')
    # if len(last_t_qs) > 0:
    #     last_t = last_t_qs[0]
    #     if last_t.action == 'TA':
    #         # 'TA'训练完即建立了新的主分支
    #         doing_tf_qs = TrainAction.objects.filter(action='TF').filter(state__lte=common.TRAIN_STATE_TRAINING).order_by('-id')
    #         if len(doing_tf_qs)>0:
    #             doing_tf = doing_tf_qs[0]
    #             if doing_tf.create_time < last_t.complete_time:
    #                 # 退出TA之前的TF
    #                 doing_tf.state = common.TRAIN_STATE_STOP
    #                 doing_tf.save()
    #         doing_tc_qs = TrainAction.objects.filter(action='TC').filter(state__lte=common.TRAIN_STATE_TRAINING).order_by('-id')
    #         if len(doing_tc_qs)>0:
    #             doing_tc = doing_tc_qs[0]
    #             if doing_tc.create_time < last_t.complete_time:
    #                 # 退出TA之前的TC
    #                 doing_tc.state = common.TRAIN_STATE_STOP
    #                 doing_tc.save()
    #
    #
    #     train_model_qs = TrainModel.objects.filter(train_action_id=last_t.pk).exclude(model_path='').order_by('-id')
    #     doing_tf_qs = TrainAction.objects.filter(action='TF').filter(state__lte=common.TRAIN_STATE_TRAINING).order_by('-id')
    #     doing_tc_qs = TrainAction.objects.filter(action='TC').filter(state__lte=common.TRAIN_STATE_TRAINING).order_by('-id')
    #     f_train_model = train_model_qs[0]
    #     f_train_upcs = last_t.upcs.all()
    #     train_upcs = TrainUpc.objects.all()
    #
    #     upcs = []
    #     for train_upc in train_upcs:
    #         upcs.append(train_upc.upc)
    #
    #     upcs = sorted(upcs)
    #
    #     f_upcs = []
    #     for train_upc in f_train_upcs:
    #         f_upcs.append(train_upc.upc)
    #
    #     # 根据train_upcs和f_train_upcs进行样本筛选
    #     append_upcs = []
    #     for upc in upcs:
    #         if upc not in f_upcs:
    #             append_upcs.append(upc)
    #
    #     train_image_qs = TrainImage.objects.filter(create_time__gt=last_t.create_time)
    #     if len(append_upcs) > 0:
    #         if len(doing_tc_qs) == 0 and len(train_image_qs) >= 10:
    #             # if len(doing_tf_qs) > 0:
    #                 # 退出正在训练的TF
    #                 # doing_tf = doing_tf_qs[0]
    #                 # doing_tf.state = 9
    #                 # doing_tf.save()
    #             logger.info('create_train: TC,新增类（{}）,新增样本（{}）'.format(len(append_upcs), len(train_image_qs)))
    #             _create_train('TC', f_train_model.pk)
    #     else:
    #         if len(doing_tf_qs) == 0:
    #             now = datetime.datetime.now()
    #             if (now - last_t.create_time).days >= 1 or len(train_image_qs) >= 200:
    #                 logger.info('create_train: TF,新增样本（{}）,间距天数（{}）'.format(len(train_image_qs),
    #                                                                         (now - last_t.create_time).days))
    #                 _create_train('TF', f_train_model.pk)
    #
    return ''


def _do_create_train_ta():
    # TA
    doing_ta = TrainAction.objects.filter(action='TA').filter(state__lte=common.TRAIN_STATE_TRAINING)
    if len(doing_ta) == 0:
        last_ta_group_qs = TrainAction.objects.filter(action='TA').filter(state=common.TRAIN_STATE_COMPLETE).values_list('deviceid').annotate(Max('create_time'))
        deviceid_exclude_qs = DeviceidExclude.objects.all().values('deviceid')
        train_image_group_qs = TrainImage.objects.exclude(deviceid__in=deviceid_exclude_qs).values_list('deviceid').annotate(cnt=Count('id')).order_by('-cnt')

        # 样本有200个
        if len(train_image_group_qs) > 0 and train_image_group_qs[0][1] >= 200:
            logger.info('create_train: TA,新增样本（{}）'.format(train_image_group_qs[0][1]))
            _create_train('TA', train_image_group_qs[0][0], None)
        # else:
        #     # 间距达到7天或者新增样本超过200个
        #     now = datetime.datetime.now()
        #     deviceid_exclude_qs = DeviceidExclude.objects.all().values('deviceid')
        #     train_image_group_qs = TrainImage.objects.filter(create_time__gt=last_ta.create_time).exclude(deviceid__in=deviceid_exclude_qs).values_list('deviceid').annotate(cnt=Count('id')).order_by('-cnt')
        #
        #     if (now - last_ta.create_time).days >= 7 or len(train_image_qs) >= 200:
        #         logger.info('create_train: TA,新增样本（{}）,间距天数（{}）'.format(len(train_image_qs),
        #                                                                 (now - last_ta.create_time).days))
        #         _create_train('TA', train_image_group_qs[0][0],None)


def _create_train(action, deviceid, f_model_id):
    train_action = TrainAction.objects.create(
        action=action,
        deviceid=deviceid,
        f_model_id=f_model_id,
        desc=''
    )

    train_action.train_path = os.path.join(common.get_train_path(), str(train_action.pk))
    # 数据准备
    names_to_labels, training_filenames, validation_filenames = convert_goods.prepare_train(train_action, deviceid)

    if names_to_labels is None:
        train_action.state = common.TRAIN_STATE_COMPLETE_WITH_ERROR
        train_action.save()
        return

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
    doing_qs = TaskLog.objects.filter(name='execute_train').filter(state=common.TASK_STATE_DOING)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='execute_train',
        ip=get_host_ip(),
    )

    try:
        logger.info('execute_train: begin task')
        ret = _do_execute_train()
        logger.info('execute_train: end task')
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
    if my_ip == '192.168.1.170':
        quit_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_STOP).exclude(action='TC')
        for quit_train in quit_train_qs:
            common.stop_train_ps(quit_train)
            quit_train.state = common.TRAIN_STATE_COMPLETE_WITH_STOP
            quit_train.complete_time = datetime.datetime.now()
            quit_train.save()

        begin_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_WAITING).exclude(action='TC')
        for begin_train in begin_train_qs:
            train_command, eval_command = _do_begin_train(begin_train)
            begin_train.ip = my_ip
            begin_train.train_command = train_command
            begin_train.eval_command = eval_command
            begin_train.state = common.TRAIN_STATE_TRAINING
            begin_train.save()
    elif my_ip == '192.168.1.73':
        quit_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_STOP).filter(action='TC')
        for quit_train in quit_train_qs:
            common.stop_train_ps(quit_train)
            quit_train.state = common.TRAIN_STATE_COMPLETE_WITH_STOP
            quit_train.complete_time = datetime.datetime.now()
            quit_train.save()
        begin_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_WAITING).filter(action='TC')
        for begin_train in begin_train_qs:
            train_command, eval_command = _do_begin_train(begin_train)
            begin_train.ip = my_ip
            begin_train.train_command = train_command
            begin_train.eval_command = eval_command
            begin_train.state = common.TRAIN_STATE_TRAINING
            begin_train.save()

    return ''


def _do_begin_train(train_action):

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
            'nasnet_mobile',
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
            'nasnet_mobile',
            1,
            8,
            '1',
            checkpoint_path,
            train_action.action
        )
    else:
        train_command = 'nohup python3 {}/goods2/dl/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  ' > /root/train_{}.out 2>&1 &'.format(
            settings.BASE_DIR,
            train_action.train_path,
            train_action.train_path,
            train_action.train_cnt,
            'nasnet_mobile',
            1,
            8,
            '0',
            train_action.action
        )
    subprocess.call(train_command, shell=True)
    # 评估
    eval_command = 'nohup python3 {}/goods2/dl/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={}' \
              ' > /root/eval_{}.out 2>&1 &'.format(
        settings.BASE_DIR,
        train_action.train_path,
        common.get_dataset_dir(True),
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
        logger.info('check_train: begin task')
        ret = _do_check_train()
        logger.info('check_train: end task')
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
    if my_ip == '192.168.1.170':
        doing_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_TRAINING).filter(ip=my_ip).exclude(action='TC')
        for check_train in doing_train_qs:
            _do_check_one_train(check_train)
    elif my_ip == '192.168.1.73':
        doing_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_TRAINING).filter(ip=my_ip).filter(action='TC')
        for check_train in doing_train_qs:
            _do_check_one_train(check_train)

    return ret

# 生成信号：
# precision达0.9后：
# TA：step超过5000且有5次eval，最后一次和前10次精度上升小于0.5%
# TF：step超过1000且有5次eval，最后一次和前10次精度上升小于1%
# TC：step超过500且有3次eval，最后一次和前5次精度上升小于1%
def _do_check_one_train(train_action):
    train_pid = common.get_train_pid(train_action)
    if train_pid == 0:
        train_action.state=common.TRAIN_STATE_COMPLETE_WITH_ERROR
        train_action.save()
        logger.error('train process has been killed:{};'.format(train_action.pk))
        return 'train process has been killed:{};'.format(train_action.pk)

    _syn_event_log(train_action)
    eval_log_qs = EvalLog.objects.filter(train_action_id=train_action.pk).order_by('-id')

    if len(eval_log_qs)>0 and eval_log_qs[0].precision>=0.9:
        last_eval_log = eval_log_qs[0]
        precision_interval = 1
        if train_action.action == 'TA':
            if len(eval_log_qs)>5:
                precision_interval = last_eval_log.precision - eval_log_qs[5].precision
            if last_eval_log.checkpoint_step>=5000 and precision_interval<=0.005:
                _do_create_train_model(train_action, last_eval_log.checkpoint_step,last_eval_log.precision)
        elif train_action.action == 'TF':
            if len(eval_log_qs)>5:
                precision_interval = last_eval_log.precision - eval_log_qs[5].precision
            if last_eval_log.checkpoint_step>=1000 and precision_interval<=0.01:
                _do_create_train_model(train_action, last_eval_log.checkpoint_step,last_eval_log.precision)
        elif train_action.action == 'TC':
            if len(eval_log_qs)>3:
                precision_interval = last_eval_log.precision - eval_log_qs[3].precision
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
        train_action.save()
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
