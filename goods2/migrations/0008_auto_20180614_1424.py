# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-06-14 14:24
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0007_auto_20180614_1400'),
    ]

    operations = [
        migrations.RenameField(
            model_name='trainaction',
            old_name='example_cnt',
            new_name='train_cnt',
        ),
        migrations.AddField(
            model_name='trainaction',
            name='validation_cnt',
            field=models.IntegerField(default=0),
        ),
    ]
