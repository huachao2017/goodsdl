from django.db import models
import datetime
from django.conf import settings

def image_upload_source(instance,filename):
    now = datetime.datetime.now()
    return '{}/{}/{}/{}_{}_{}'.format(settings.DETECT_DIR_NAME, instance.deviceid, now.strftime('%Y%m/%d%H'), now.strftime('%M%S'), str(now.time()), filename)

class Image(models.Model):
    deviceid = models.CharField(max_length=20, default='0',db_index=True)
    ret = models.CharField(max_length=500, default='')
    source = models.ImageField(max_length=200, upload_to=image_upload_source)
    aiinterval = models.FloatField(default=0.0)
    lastinterval = models.FloatField(default=0.0)
    create_time = models.DateTimeField('date created', auto_now_add=True,db_index=True)

class ImageClass(models.Model):
    deviceid = models.CharField(max_length=20, default='0')
    source = models.ImageField(max_length=200, upload_to=image_upload_source)
    create_time = models.DateTimeField('date created', auto_now_add=True)

class GoodsClass(models.Model):
    image_class = models.ForeignKey(ImageClass,related_name="image_class_goods",on_delete=models.CASCADE)
    class_type = models.IntegerField(default=0)
    score = models.FloatField(default=0.0)
    upc = models.CharField(max_length=20)

class Goods(models.Model):
    image = models.ForeignKey(Image,related_name="image_goods",on_delete=models.CASCADE)
    class_type = models.IntegerField(default=0)
    score1 = models.FloatField(default=0.0)
    score2 = models.FloatField(default=0.0)
    upc = models.CharField(max_length=20)
    xmin = models.PositiveIntegerField(default=0)
    ymin = models.PositiveIntegerField(default=0)
    xmax = models.PositiveIntegerField(default=0)
    ymax = models.PositiveIntegerField(default=0)

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
    ret = '{}/data_{}/{}/{}_{}'.format(settings.DATASET_DIR_NAME, instance.deviceid, instance.upc, str(now.time()), filename)
    return ret

def train_image_class_upload_source(instance,filename):
    now = datetime.datetime.now()
    ret = '{}/data_new_{}/{}/{}_{}'.format(settings.DATASET_DIR_NAME, instance.deviceid, instance.upc, str(now.time()),
                                  filename)
    return ret

def train_image_only_upload_source(instance,filename):
    now = datetime.datetime.now()
    ret = '{}/data_raw_{}/{}/{}_{}'.format(settings.DATASET_DIR_NAME, instance.deviceid, now.strftime('%Y%m%d'), str(now.time()), filename)
    return ret

class TrainImage(models.Model):
    deviceid = models.CharField(max_length=20, default='')
    traintype = models.PositiveIntegerField(default=0)
    source = models.ImageField(max_length=200, upload_to=train_image_upload_source)
    upc = models.CharField(max_length=20)
    name = models.CharField(max_length=128, default='')
    xmin = models.PositiveIntegerField(default=0)
    ymin = models.PositiveIntegerField(default=0)
    xmax = models.PositiveIntegerField(default=0)
    ymax = models.PositiveIntegerField(default=0)
    create_time = models.DateTimeField('date created', auto_now_add=True)

class TrainImageOnly(models.Model):
    deviceid = models.CharField(max_length=20, default='')
    traintype = models.PositiveIntegerField(default=0)
    source = models.ImageField(max_length=200, upload_to=train_image_only_upload_source)
    create_time = models.DateTimeField('date created', auto_now_add=True)

class TrainImageClass(models.Model):
    deviceid = models.CharField(max_length=20, default='')
    traintype = models.PositiveIntegerField(default=0)
    source = models.ImageField(max_length=200, upload_to=train_image_class_upload_source)
    upc = models.CharField(max_length=20)
    name = models.CharField(max_length=20, default='')
    create_time = models.DateTimeField('date created', auto_now_add=True)

class DatasetAction(models.Model):
    ACTION_CHOICES = (
        (u'D2', u'Dataset Step 2'),
    )
    action = models.CharField(max_length=2, choices=ACTION_CHOICES)
    traintype = models.PositiveIntegerField(default=0)
    desc = models.CharField(max_length=500,null=True)
    param = models.CharField(max_length=500)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)
    def __str__(self):
        return '{}:{}:{}'.format(self.action, self.pk, self.desc)

class TrainAction(models.Model):
    ACTION_CHOICES = (
        (u'T1', u'Train Step 1'),
        (u'T2', u'Train Step 2'),
        (u'T3', u'Train Step 3'),
        (u'TC', u'Train Only Step 2'),
    )
    action = models.CharField(max_length=2, choices=ACTION_CHOICES)
    traintype = models.PositiveIntegerField(default=0) # use for step3
    is_fineture = models.BooleanField(default=True)
    MODEL_CHOICES = (
        (u'ANY', u'not specify'),
        (u'inception_resnet_v2', u'inception resnet V2'),
        (u'nasnet_large', u'nas large'),
        (u'nasnet_mobile', u'nas mobile'),
    )
    model_name = models.CharField(max_length=50, choices=MODEL_CHOICES, default='ANY')
    desc = models.CharField(max_length=500,null=True)
    param = models.CharField(max_length=500,null=True)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)
    def __str__(self):
        return '{}:{}:{}'.format(self.action, self.pk, self.desc)

class ExportAction(models.Model):
    train_action = models.ForeignKey(TrainAction,related_name="export_actions",on_delete=models.CASCADE)
    checkpoint_prefix = models.PositiveIntegerField(default=0)
    MODEL_CHOICES = (
        (u'ANY', u'not specify'),
        (u'inception_resnet_v2', u'inception resnet V2'),
        (u'nasnet_large', u'nas large'),
        (u'nasnet_mobile', u'nas mobile'),
    )
    model_name = models.CharField(max_length=50, choices=MODEL_CHOICES, default='ANY')
    param = models.CharField(max_length=500)
    create_time = models.DateTimeField('date created', auto_now_add=True)
    update_time = models.DateTimeField('date updated', auto_now=True)

class StopTrainAction(models.Model):
    train_action = models.ForeignKey(TrainAction,related_name="stop_train_actions",on_delete=models.CASCADE)
    param = models.CharField(max_length=500)
    create_time = models.DateTimeField('date created', auto_now_add=True)

class RfidImageCompareAction(models.Model):
    deviceid = models.CharField(max_length=20, default='0')
    startTime = models.DateField('start time')
    endTime = models.DateField('end time')

class RfidTransaction(models.Model):
    image = models.ForeignKey(Image, related_name="rfid_transaction_image",on_delete=models.CASCADE,null=True)
    transaction_time = models.DateTimeField('transaction date',db_index=True)
    create_time = models.DateTimeField('date created', auto_now_add=True)

class RfidGoods(models.Model):
    rfid_transaction = models.ForeignKey(RfidTransaction,related_name="rfid_goods",on_delete=models.CASCADE)
    upc = models.CharField(max_length=20)
    count = models.PositiveIntegerField(default=1)

class TransactionMetrix(models.Model):
    rfid_transaction = models.OneToOneField(RfidTransaction,related_name="transaction_metrix",on_delete=models.CASCADE)
    same_upc_num = models.PositiveIntegerField(default=0)
    only_rfid_upc_num = models.PositiveIntegerField(default=0)
    only_image_upc_num = models.PositiveIntegerField(default=0)

