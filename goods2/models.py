from django.db import models
import datetime
from . import common

def image_upload_source(instance, filename):
    now = datetime.datetime.now()
    return '{}/{}/{}/{}/{}_{}_{}'.format(common.DETECT_DIR, instance.deviceid, now.strftime('%Y%m'),
                                                now.strftime('%d%H'), now.strftime('%M%S'), str(now.time()), filename)


class ImageGroundTruth(models.Model):
    identify = models.CharField(max_length=64, db_index=True)
    groundtruth_upc = models.CharField(max_length=20, db_index=True)
    create_time = models.DateTimeField('date created', auto_now_add=True, db_index=True)


class Image(models.Model):
    image_ground_truth = models.ForeignKey(ImageGroundTruth, related_name="ground_truth_images", default=None)
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
    ret = '{}/{}/{}_{}_{}'.format(common.DATASET_DIR_NAME, instance.upc, instance.deviceid, str(now.time()),
                                         filename)
    return ret


class TrainImage(models.Model):
    deviceid = models.CharField(max_length=20, default='')
    source = models.ImageField(max_length=200, upload_to=train_image_upload_source)
    upc = models.CharField(max_length=20, db_index=True)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    FROM_CHOICES = (
        (1, u'backend'),
        (2, u'frontend'),
    )
    source_from = models.IntegerField(choices=FROM_CHOICES, default=1)


class TrainUpc(models.Model):
    upc = models.CharField(max_length=20, unique=True)
    cnt = models.IntegerField(default=1)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)


MODEL_CHOICES = (
    (u'nasnet_large', u'nas large'),
    (u'nasnet_mobile', u'nas mobile'),
    (u'inception_resnet_v2', u'inception resnet V2'),
)


class TrainAction(models.Model):
    train_ip = models.GenericIPAddressField()
    ACTION_CHOICES = (
        (u'TA', u'Train All Dataset'),
        (u'TF', u'Train Funiture'),
        (u'TC', u'Train Add Class'),
    )
    action = models.CharField(max_length=5, choices=ACTION_CHOICES)
    train_path = models.CharField(max_length=200)
    model_name = models.CharField(max_length=50, choices=MODEL_CHOICES, default='nasnet_large')
    STATE_CHOICES = (
        (1, u'Waiting'),
        (5, u'Training'),
        (9, u'pause'),
        (10, u'Complete'),
        (20, u'Quit'),
    )
    state = models.IntegerField(choices=STATE_CHOICES, default=1)
    max_step = models.IntegerField(default=100000)
    desc = models.CharField(max_length=500, null=True)

    # TF and TC must have f_model
    f_model = models.ForeignKey('TrainModel', related_name="child_trains", default=None)

    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)

    def __str__(self):
        return '{}-{}:{}'.format(self.pk, self.action, self.desc)


class TrainModel(models.Model):
    train_action = models.ForeignKey(TrainAction, related_name="train_models", on_delete=models.CASCADE)
    model_path = models.CharField(max_length=200)
    checkpoint_prefix = models.PositiveIntegerField(default=0)
    precision = models.FloatField(default=.0)
    create_time = models.DateTimeField('date created', auto_now_add=True)

    def __str__(self):
        return '{}-{}:{}'.format(self.pk, self.train_action, self.checkpoint_prefix)


class TrainActionUpcs(models.Model):
    train_action = models.ForeignKey(TrainAction, related_name="upcs", on_delete=models.CASCADE)
    train_upc = models.ForeignKey(TrainUpc, related_name="trains", on_delete=models.CASCADE)

class TrainActionDevices(models.Model):
    train_action = models.ForeignKey(TrainAction, related_name="devices", on_delete=models.CASCADE)
    train_deviceid = models.CharField(max_length=20, default='')
