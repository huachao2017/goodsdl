from django.db import models
import datetime
from goods2 import common


def image_upload_source(instance, filename):
    now = datetime.datetime.now()
    return '{}/{}/{}/{}/{}_{}_{}'.format(common.DETECT_DIR, instance.deviceid, now.strftime('%Y%m'),
                                         now.strftime('%d%H'), now.strftime('%M%S'), str(now.time()), filename)


class Deviceid(models.Model):
    deviceid = models.CharField(max_length=20, default='0', unique=True)
    STATE_CHOICES = (
        (0, u'testing'),
        (10, u'commercial'),
    )
    state = models.IntegerField(choices=STATE_CHOICES, default=0)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)
    commercial_time = models.DateTimeField('testing finish time', auto_now_add=True)


class DeviceidPrecision(models.Model):
    device = models.ForeignKey(Deviceid, related_name="device_precisions", on_delete=models.CASCADE)
    truth_rate = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    create_time = models.DateTimeField('date created', auto_now_add=True)


class ImageGroundTruth(models.Model):
    deviceid = models.CharField(max_length=20, default='0', db_index=True)
    identify = models.CharField(max_length=64, db_index=True)
    upc = models.CharField(max_length=20, db_index=True)
    cnt = models.IntegerField(default=0)
    truth_rate = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    create_time = models.DateTimeField('date created', auto_now_add=True, db_index=True)


class Image(models.Model):
    image_ground_truth = models.ForeignKey(ImageGroundTruth, related_name="ground_truth_images", default=None,
                                           null=True, on_delete=models.SET_NULL)
    deviceid = models.CharField(max_length=20, default='0', db_index=True)
    identify = models.CharField(max_length=64, db_index=True)
    source = models.ImageField(max_length=200, upload_to=image_upload_source)
    create_time = models.DateTimeField('date created', auto_now_add=True, db_index=True)


class ImageResult(models.Model):
    image = models.ForeignKey(Image, related_name="image_results", on_delete=models.CASCADE)
    upc = models.CharField(max_length=20)
    score = models.FloatField(default=0.0)


def train_image_upload_source(instance, filename):
    now = datetime.datetime.now()
    ret = '{}/{}/{}_{}_{}'.format(common.DATASET_DIR, instance.upc, instance.deviceid, str(now.time()),
                                  filename)
    return ret


class TrainImage(models.Model):
    deviceid = models.CharField(max_length=20, default='')
    source = models.ImageField(max_length=200, upload_to=train_image_upload_source)
    upc = models.CharField(max_length=20, db_index=True)
    source_image = models.ForeignKey('Image', related_name="trainimages", default=None, null=True,
                                     on_delete=models.SET_NULL)
    FROM_CHOICES = (
        (1, u'backend'),
        (2, u'frontend'),
    )
    source_from = models.IntegerField(choices=FROM_CHOICES, default=1)
    score = models.FloatField(default=1.0)  # 可靠度评分，用于筛选样本
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)


class TrainUpc(models.Model):
    upc = models.CharField(max_length=20, unique=True)
    cnt = models.IntegerField(default=1)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)


class TrainAction(models.Model):
    ACTION_CHOICES = (
        (u'TA', u'Train All Dataset'),
        (u'TF', u'Train Funiture'),
        (u'TC', u'Train Add Class'),
    )
    action = models.CharField(max_length=5, choices=ACTION_CHOICES)
    train_path = models.CharField(max_length=200)
    STATE_CHOICES = (
        (1, u'Waiting'),
        (5, u'Training'),
        (9, u'Stop'),
        (10, u'Complete'),
        (20, u'Complete with stop'),
    )
    state = models.IntegerField(choices=STATE_CHOICES, default=1)
    train_cnt = models.IntegerField(default=0)
    validation_cnt = models.IntegerField(default=0)
    desc = models.CharField(max_length=500, null=True)

    # TF and TC must have f_model
    f_model = models.ForeignKey('TrainModel', related_name="child_trains", default=None, null=True,
                                on_delete=models.SET_NULL)

    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)
    complete_time = models.DateTimeField('train finish time', auto_now_add=True)

    def __str__(self):
        return '{}-{}:{}'.format(self.pk, self.action, self.desc)


class EvalLog(models.Model):
    train_action = models.ForeignKey(TrainAction, related_name="train_evals", on_delete=models.CASCADE)
    precision = models.FloatField(default=0.0)
    checkpoint_step = models.IntegerField(default=0)
    create_time = models.DateTimeField('date created', auto_now_add=True)


class TrainModel(models.Model):
    train_action = models.ForeignKey(TrainAction, related_name="train_models", on_delete=models.CASCADE)
    model_path = models.CharField(max_length=200, default='')
    checkpoint_step = models.PositiveIntegerField(default=0)
    precision = models.FloatField(default=.0)
    create_time = models.DateTimeField('date created', auto_now_add=True)

    def __str__(self):
        return '{}-{}:{},{}'.format(self.pk, self.train_action, self.checkpoint_step, self.precision)


class ImageTrainModel(models.Model):
    train_model = models.ForeignKey(TrainModel, related_name="images", on_delete=models.CASCADE)
    image = models.ForeignKey(Image, related_name="train_models", on_delete=models.CASCADE)


class TrainActionUpcs(models.Model):
    train_action = models.ForeignKey(TrainAction, related_name="upcs", on_delete=models.CASCADE)
    train_upc = models.ForeignKey(TrainUpc, related_name="trains", on_delete=models.CASCADE)
    upc = models.CharField(max_length=20, db_index=True)
    cnt = models.IntegerField(default=1)


class TrainActionDevices(models.Model):
    train_action = models.ForeignKey(TrainAction, related_name="devices", on_delete=models.CASCADE)
    train_deviceid = models.CharField(max_length=20, default='')


class TaskLog(models.Model):
    name = models.CharField(max_length=50)
    ip = models.CharField(max_length=50, default='')
    STATE_CHOICES = (
        (1, u'Doing'),
        (10, u'Complete'),
        (20, u'Error'),
    )
    message = models.CharField(max_length=500, default='')
    state = models.IntegerField(choices=STATE_CHOICES, default=1)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)
