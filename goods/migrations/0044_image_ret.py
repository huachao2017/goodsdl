# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-03-14 14:40
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0043_auto_20180309_1106'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='ret',
            field=models.CharField(default='', max_length=500),
        ),
    ]
