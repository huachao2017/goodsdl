# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-08-10 15:24
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0032_auto_20180810_1457'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainimage',
            name='deviceid',
            field=models.CharField(db_index=True, default='', max_length=20),
        ),
    ]