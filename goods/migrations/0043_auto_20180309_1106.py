# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-03-09 11:06
from __future__ import unicode_literals

from django.db import migrations, models
import goods.models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0042_auto_20180202_1902'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainaction',
            name='model_name',
            field=models.CharField(choices=[('ANY', 'not specify'), ('inception_resnet_v2', 'inception resnet V2'), ('nasnet_large', 'nas large'), ('nasnet_mobile', 'nas mobile')], default='ANY', max_length=50),
        ),
        migrations.AlterField(
            model_name='exportaction',
            name='model_name',
            field=models.CharField(choices=[('ANY', 'not specify'), ('inception_resnet_v2', 'inception resnet V2'), ('nasnet_large', 'nas large'), ('nasnet_mobile', 'nas mobile')], default='ANY', max_length=50),
        ),
        migrations.AlterField(
            model_name='trainaction',
            name='action',
            field=models.CharField(choices=[('T1', 'Train Step 1'), ('T2', 'Train Step 2'), ('T3', 'Train Step 3'), ('TC', 'Train Only Step 2')], max_length=2),
        ),
        migrations.AlterField(
            model_name='trainaction',
            name='param',
            field=models.CharField(max_length=500, null=True),
        ),
        migrations.AlterField(
            model_name='trainimageclass',
            name='source',
            field=models.ImageField(max_length=200, upload_to=goods.models.train_image_class_upload_source),
        ),
    ]