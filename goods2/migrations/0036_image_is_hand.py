# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-10-16 13:51
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0035_image_is_train'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='is_hand',
            field=models.BooleanField(default=False),
        ),
    ]