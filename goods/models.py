from django.db import models
import datetime

def image_upload_source(instance,filename):
    now = datetime.datetime.now()
    subdir = now.strftime('%Y%m/%d%H/%M%S')
    return '{}_{}_{}'.format(subdir, str(now.time()), filename)

class Image(models.Model):
    source = models.ImageField(max_length=200, upload_to=image_upload_source)
    create_time = models.DateTimeField('date created', auto_now_add=True)

class Goods(models.Model):
    image = models.ForeignKey(Image,related_name="image_goods",on_delete=models.CASCADE)
    class_type = models.IntegerField(default=0)
    score = models.FloatField(default=0.0)
    upc = models.CharField(max_length=20)
    xmin = models.PositiveIntegerField(default=0)
    ymin = models.PositiveIntegerField(default=0)
    xmax = models.PositiveIntegerField(default=0)
    ymax = models.PositiveIntegerField(default=0)

def train_image_upload_source(instance,filename):
    now = datetime.datetime.now()
    return 'data/{}/{}_{}'.format(instance.upc, str(now.time()), filename)

class TrainImage(models.Model):
    deviceid = models.CharField(max_length=20, default='')
    source = models.ImageField(max_length=200, upload_to=train_image_upload_source)
    upc = models.CharField(max_length=20)
    name = models.CharField(max_length=20, default='')
    xmin = models.PositiveIntegerField(default=0)
    ymin = models.PositiveIntegerField(default=0)
    xmax = models.PositiveIntegerField(default=0)
    ymax = models.PositiveIntegerField(default=0)
    create_time = models.DateTimeField('date created', auto_now_add=True)

class ActionLog(models.Model):
    ACTION_CHOICES = (
        (u'BT', u'Begin Train'),
        (u'ST', u'Stop Train'),
        (u'EG', u'Export Graph'),
    )
    action = models.CharField(max_length=2, choices=ACTION_CHOICES)
    desc = models.CharField(max_length=500)
    param = models.CharField(max_length=500)
    create_time = models.DateTimeField('date created', auto_now_add=True)
