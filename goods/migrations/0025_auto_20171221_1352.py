# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-12-21 13:52
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0024_auto_20171221_1346'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rfidimagecompareaction',
            name='endTime',
            field=models.DateField(verbose_name='end time'),
        ),
        migrations.AlterField(
            model_name='rfidimagecompareaction',
            name='startTime',
            field=models.DateField(verbose_name='start time'),
        ),
    ]
