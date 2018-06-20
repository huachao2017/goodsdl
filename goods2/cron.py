import logging
import os
import shutil
from django.conf import settings
import tensorflow as tf
from .models import Image, ImageResult, ImageGroundTruth, TrainImage, TrainUpc, TaskLog, TrainAction, TrainActionUpcs, \
    TrainModel, EvalLog
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
        ip=get_host_ip(),
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
        ip=get_host_ip()
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
        train_model_qs = TrainModel.objects.filter(train_action_id=last_t.pk).exclude(model_path='').order_by('-id')
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
                # if len(doing_tf_qs) > 0:
                    # 退出正在训练的TF
                    # doing_tf = doing_tf_qs[0]
                    # doing_tf.state = 9
                    # doing_tf.save()
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
        ip=get_host_ip(),
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
            _do_stop_train(quit_train)
            quit_train.state = 20
            quit_train.save()

        begin_train_qs = TrainAction.objects.filter(state=1).exclude(action='TC')
        for begin_train in begin_train_qs:
            _do_begin_train(begin_train)
            begin_train.state = 5
            begin_train.save()
    elif my_ip == '192.168.1.173':
        quit_train_qs = TrainAction.objects.filter(state=9).filter(action='TC')
        for quit_train in quit_train_qs:
            _do_stop_train(quit_train)
            quit_train.state = 20
            quit_train.save()
        begin_train_qs = TrainAction.objects.filter(state=1).filter(action='TC')
        for begin_train in begin_train_qs:
            _do_begin_train(begin_train)
            begin_train.state = 5
            begin_train.save()

    return ''


def _do_begin_train(train_action):
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
        command = 'nohup python3 {}/goods2/dl/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  '--checkpoint_path={} > /root/train_{}.out 2>&1 &'.format(
            settings.BASE_DIR,
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
        command = 'nohup python3 {}/goods2/dl/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --CUDA_VISIBLE_DEVICES={}' \
                  ' > /root/train_{}.out 2>&1 &'.format(
            settings.BASE_DIR,
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
    command = 'nohup python3 {}/goods2/dl/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={}' \
              ' > /root/eval_{}.out 2>&1 &'.format(
        settings.BASE_DIR,
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


def _do_stop_train(train_action):
    train_dir = os.path.join(common.TRAIN_DIR, str(train_action.pk))
    train_ps = os.popen('ps -ef | grep train.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if train_ps != '':
        pid = int(train_ps.split()[1])
        os.system('kill -s 9 {}'.format(str(pid)))
    eval_ps = os.popen('ps -ef | grep eval2.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if eval_ps != '':
        pid = int(eval_ps.split()[1])
        os.system('kill -s 9 {}'.format(str(pid)))


def check_train():
    doing_qs = TaskLog.objects.filter(name='check_train').filter(state=1)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='check_train',
        ip=get_host_ip(),
    )

    try:
        logger.info('execute_train: begin task')
        ret = _do_check_train()
        logger.info('execute_train: end task')
    except Exception as e:
        cur_task.state = 20
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state = 10
        cur_task.message = ret
        cur_task.save()

def _do_check_train():
    my_ip = get_host_ip()
    ret = ''
    if my_ip == '192.168.1.170':
        doing_train_qs = TrainAction.objects.filter(state=5).exclude(action='TC')
        for check_train in doing_train_qs:
            ret += _do_check_one_train(check_train)
    elif my_ip == '192.168.1.173':
        doing_train_qs = TrainAction.objects.filter(state=5).filter(action='TC')
        for check_train in doing_train_qs:
            ret += _do_check_one_train(check_train)

    return ret

# 生成信号：
# precsion达0.95后：
# TA：距离上次生成超5000/step或精度上升超0.3%存一次
# TF：距离上次生成超2000/step或精度上升超0.6%存一次
# TC：距离上次生成超500/step或精度上升超1%存一次
#
# 训练结束信号（导出）：
# precsion达0.95后：
# TA：10次生成达0.998以上或精度提升小于0.2%后结束
# TF：5次生成达0.995以上或精度提升小于0.3%后结束
# TC：3次生成达0.99以上或精度提升小于0.5%后结束
def _do_check_one_train(train_action):
    train_ps = os.popen('ps -ef | grep train.py | grep {} | grep -v grep'.format(train_action.train_path)).readline()
    if train_ps == '':
        train_action.state=20
        train_action.save()
        logger.error('train process has been killed:{};'.format(train_action.pk))
        return  'train process has been killed:{};'.format(train_action.pk)

    eval_log_qs = EvalLog.objects.filter(train_action_id=train_action.pk).order_by('-id')

    train_model_qs = TrainModel.objects.filter(train_action_id=train_action.pk).order_by('-id')
    last_train_model = None
    if len(train_model_qs)>0:
        last_train_model = train_model_qs[0]

    if len(eval_log_qs)>0 and eval_log_qs[0].precision>0.95:
        if last_train_model is None:
            _do_create_train_model(train_action, eval_log_qs[0])
        else:
            step_interval = eval_log_qs[0].checkpoint_step - last_train_model.checkpoint_step
            precision_interval = eval_log_qs[0].precision - last_train_model.precision
            if train_action.action == 'TA':
                if step_interval>=5000 or precision_interval>=0.003:
                    _do_create_train_model(train_action,eval_log_qs[0])
            elif train_action.action == 'TF':
                if step_interval>=2000 or precision_interval>=0.006:
                    _do_create_train_model(train_action,eval_log_qs[0])
            elif train_action.action == 'TC':
                if step_interval>=500 or precision_interval>=0.01:
                    _do_create_train_model(train_action,eval_log_qs[0])


def _do_create_train_model(train_action, eval_log):
    cur_train_model = TrainModel.objects.create(
        train_action_id=train_action.pk,
        checkpoint_step=eval_log.checkpoint_step,
        precision=eval_log.precision,
    )
    if train_action.action == 'TA':
        count = 10
    elif train_action.action == 'TF':
        count = 5
    elif train_action.action == 'TC':
        count = 3
    train_model_qs = TrainModel.objects.filter(train_action_id=train_action.pk).order_by('-id')[:count]
    if len(train_model_qs) == count:
        min_precision = train_model_qs[0].precision
        max_precision = train_model_qs[0].precision
        for train_model in train_model_qs:
            if min_precision > train_model.precision:
                min_precision = train_model.precision
            if max_precision < train_model.precision:
                max_precision = train_model.precision
        if train_action.action == 'TA':
            if min_precision > 0.998 or max_precision - min_precision < 0.002:
                _do_export_train(train_action, cur_train_model)
        elif train_action.action == 'TF':
            if min_precision > 0.995 or max_precision - min_precision < 0.003:
                _do_export_train(train_action, cur_train_model)
        elif train_action.action == 'TC':
            if min_precision > 0.99 or max_precision - min_precision < 0.005:
                _do_export_train(train_action, cur_train_model)
    return cur_train_model


def _do_export_train(train_action, train_model):
    logger.info('Export <trainid:{}> Graph from classify train.'.format(train_action.pk))
    checkpoint_model_path = tf.train.latest_checkpoint(train_action.train_path)
    if checkpoint_model_path:
        checkpoint_step = checkpoint_model_path.split('-')[-1]
        model_path = os.path.join(common.MODEL_DIR, str(train_model.pk))
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
        # copy dataset
        # shutil.copy(os.path.join(settings.TRAIN_ROOT, str(train_action.pk), 'goods_recogonize_train.tfrecord'),
        #             model_dir)
        # copy label
        shutil.copy(os.path.join(train_action.train_path, 'labels.txt'), model_path)

        train_model.checkpoint_step = checkpoint_step
        train_model.model_path = model_path
        train_model.save()
        train_action.state = 10
        train_action.save()

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip
