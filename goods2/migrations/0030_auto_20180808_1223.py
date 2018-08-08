# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-08-08 12:23
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0029_auto_20180628_1755'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='trainactiondevices',
            name='train_action',
        ),
        migrations.AddField(
            model_name='trainaction',
            name='deviceid',
            field=models.CharField(db_index=True, default='', max_length=20),
        ),
        migrations.AddField(
            model_name='trainupc',
            name='deviceid',
            field=models.CharField(db_index=True, default='', max_length=20),
        ),
        migrations.AlterField(
            model_name='trainupc',
            name='upc',
            field=models.CharField(db_index=True, max_length=20),
        ),
        migrations.DeleteModel(
            name='TrainActionDevices',
        ),
    ]