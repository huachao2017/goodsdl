from django.db import models
import datetime

def image_upload_source(instance,filename):
    now = datetime.datetime.now()
    return '{}/{}/{}_{}_{}'.format(now.strftime('%Y%m/%d%H'), instance.deviceid, now.strftime('%M%S'), str(now.time()), filename)

class Image(models.Model):
    deviceid = models.CharField(max_length=20, default='0')
    source = models.ImageField(max_length=200, upload_to=image_upload_source)
    create_time = models.DateTimeField('date created', auto_now_add=True)

# class Goods(models.Model):
#     image = models.ForeignKey(Image,related_name="image_goods",on_delete=models.CASCADE)
#     class_type = models.IntegerField(default=0)
#     score = models.FloatField(default=0.0)
#     upc = models.CharField(max_length=20)
#     xmin = models.PositiveIntegerField(default=0)
#     ymin = models.PositiveIntegerField(default=0)
#     xmax = models.PositiveIntegerField(default=0)
#     ymax = models.PositiveIntegerField(default=0)

class ProblemGoods(models.Model):
    image = models.ForeignKey(Image, related_name="image_problem_goods",on_delete=models.CASCADE)
    index = models.IntegerField(default=0)
    class_type_0 = models.IntegerField(default=0)
    class_type_1 = models.IntegerField(default=0)
    class_type_2 = models.IntegerField(default=0)
    class_type_3 = models.IntegerField(default=0)
    class_type_4 = models.IntegerField(default=0)
    score_0 = models.FloatField(default=0.0)
    score_1 = models.FloatField(default=0.0)
    score_2 = models.FloatField(default=0.0)
    score_3 = models.FloatField(default=0.0)
    score_4 = models.FloatField(default=0.0)

def train_image_upload_source(instance,filename):
    now = datetime.datetime.now()
    if instance.traintype == 0:
        ret = 'data/{}/{}_{}'.format(instance.upc, str(now.time()), filename)
    else:
        ret = '{}/{}/{}_{}'.format(instance.traintype, instance.upc, str(now.time()), filename)
    return ret

class TrainImage(models.Model):
    deviceid = models.CharField(max_length=20, default='')
    traintype = models.PositiveIntegerField(default=0)
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
        (u'T1', u'Train Step 1'),
        (u'T2', u'Train Step 2'),
        (u'ST', u'Stop Train'),
        (u'E1', u'Export Step 1 Graph'),
        (u'E2', u'Export Step 2 Graph'),
    )
    action = models.CharField(max_length=2, choices=ACTION_CHOICES)
    traintype = models.PositiveIntegerField(default=0)
    desc = models.CharField(max_length=500)
    param = models.CharField(max_length=500)
    create_time = models.DateTimeField('date created', auto_now_add=True)
