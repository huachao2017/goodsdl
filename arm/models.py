from django.db import models
import datetime
from arm import common

# Create your models here.

def image_upload_source(instance, filename):
    now = datetime.datetime.now()
    return '{}/{}/{}/{}_{}_{}'.format(common.get_detect_dir(), now.strftime('%Y%m'),
                                         now.strftime('%d%H'), now.strftime('%M%S'), str(now.time()), filename)


def train_image_upload_source(instance, filename):
    now = datetime.datetime.now()
    return '{}/{}/{}/{}_{}_{}'.format(common.get_dataset_dir(), now.strftime('%Y%m'),
                                      now.strftime('%d%H'), now.strftime('%M%S'), str(now.time()), filename)


class ArmImage(models.Model):
    rgb_source = models.ImageField(max_length=200, upload_to=image_upload_source)
    depth_source = models.ImageField(max_length=200, upload_to=image_upload_source)
    table_z = models.IntegerField(default=0)
    result = models.TextField()
    create_time = models.DateTimeField('date created', auto_now_add=True,db_index=True)


class ArmCylinderImage(models.Model):
    rgb_source = models.ImageField(max_length=200, upload_to=image_upload_source)
    depth_source = models.ImageField(max_length=200, upload_to=image_upload_source)
    table_x = models.IntegerField(default=0)
    table_y = models.IntegerField(default=0)
    table_z = models.IntegerField(default=0)
    result = models.TextField()
    create_time = models.DateTimeField('date created', auto_now_add=True, db_index=True)


class ArmTrainImage(models.Model):
    deviceid = models.CharField(max_length=20, default='100000')
    rgb_source = models.ImageField(max_length=200, upload_to=train_image_upload_source)
    depth_source = models.ImageField(max_length=200, upload_to=train_image_upload_source)
    table_z = models.IntegerField(default=0)
    upc = models.CharField(max_length=20, db_index=True)
    create_time = models.DateTimeField('date created', auto_now_add=True, db_index=True)
