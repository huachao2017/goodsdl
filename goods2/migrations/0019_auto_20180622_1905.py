# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-06-22 19:05
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0018_imagetrainmodel'),
    ]

    operations = [
        migrations.RenameField(
            model_name='imagegroundtruth',
            old_name='groundtruth_upc',
            new_name='upc',
        ),
    ]
