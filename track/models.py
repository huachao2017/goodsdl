from django.db import models

# Create your models here.

class BasePoint(models.Model):
  shopid = models.IntegerField(default=0, db_index=True)
  deviceid = models.CharField(max_length=20, db_index=True)
  x = models.IntegerField(default=0)
  y = models.IntegerField(default=0)
  z = models.IntegerField(default=0)
  create_time = models.DateTimeField('date created', auto_now_add=True)

class EntranceArea(models.Model):
  shopid = models.IntegerField(default=0, db_index=True)
  deviceid = models.CharField(max_length=20, db_index=True)
  x = models.IntegerField(default=0)
  y = models.IntegerField(default=0)
  w = models.IntegerField(default=0)
  h = models.IntegerField(default=0)
  create_time = models.DateTimeField('date created', auto_now_add=True)

class Recognition(models.Model):
  shopid = models.IntegerField(default=0, db_index=True)
  deviceid = models.CharField(max_length=20, db_index=True)
  cid = models.IntegerField(db_index=True)
  sid = models.IntegerField(db_index=True)
  r_sid = models.IntegerField(default=0)
  x = models.IntegerField(default=0)
  y = models.IntegerField(default=0)
  z = models.IntegerField(default=0)
  create_time = models.DateTimeField('date created', auto_now_add=True)


class TransferRecognition(models.Model):
  recognition = models.ForeignKey(Recognition, related_name="transfer_recognition", on_delete=models.CASCADE)
  transfer_deviceid = models.CharField(max_length=20, db_index=True)
  distance = models.IntegerField(default=0)
  create_time = models.DateTimeField('date created', auto_now_add=True)

# class RecognitionStatus(models.Model):
#   shopid = models.IntegerField(default=0, db_index=True)
#   deviceid = models.CharField(max_length=20, db_index=True)
#   cid = models.IntegerField(db_index=True)
#   STATUS_CHOICES = (
#     (0, u'lock'),
#     (1, u'finish'),
#   )
#   status = models.IntegerField(choices=STATUS_CHOICES, default=0)
#   create_time = models.DateTimeField('date created', auto_now_add=True)
#   update_time = models.DateTimeField('date updated', auto_now=True)
