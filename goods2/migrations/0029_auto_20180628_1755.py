# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-06-28 18:00
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0028_auto_20180628_1506'),
    ]

    operations = [
        migrations.AlterField(
            model_name='evallog',
            name='create_time',
            field=models.DateTimeField(verbose_name='date created'),
        ),
    ]
