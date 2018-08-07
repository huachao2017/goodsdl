from django.db import models
import datetime

# Create your models here.

class Image(models.Model):
    picurl = models.CharField(max_length=200)
    image_path = models.CharField(max_length=200, default='')
    index = models.TextField()
    create_time = models.DateTimeField('date created', auto_now_add=True,db_index=True)
