# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-09-29 19:10
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0034_image_upc'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='is_train',
            field=models.BooleanField(default=False),
        ),
    ]
