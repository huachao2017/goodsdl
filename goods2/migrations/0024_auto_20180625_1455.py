# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-06-25 14:55
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0023_auto_20180625_1429'),
    ]

    operations = [
        migrations.AddField(
            model_name='deviceidprecision',
            name='truth_rate',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='imagegroundtruth',
            name='truth_rate',
            field=models.FloatField(default=0.0),
        ),
    ]
