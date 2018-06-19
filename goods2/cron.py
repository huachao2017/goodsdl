import logging
import os
import shutil
from django.conf import settings
import tensorflow as tf
from .models import Image, ImageResult, ImageGroundTruth, TrainImage, TrainUpc, TaskLog
from . import common

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
        ret = _do_transfer_sample()
    except Exception as e:
        cur_task.state = 20
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state=10
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


def execute_train():
    doing_qs = TaskLog.objects.filter(name='execute_train').filter(state=1)
    if len(doing_qs) > 0:
        return
    cur_task = TaskLog.objects.create(
        name='execute_train',
    )

    try:
        _do_execute_train()
    except Exception as e:
        cur_task.state = 20
        cur_task.message = e
        cur_task.save()
    else:
        cur_task.state=10
        cur_task.save()

def _do_execute_train():
    pass
