# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-11-15 11:33
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0006_trainimage_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainimage',
            name='deviceid',
            field=models.CharField(default='', max_length=20),
        ),
    ]