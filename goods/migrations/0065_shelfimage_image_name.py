# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2019-05-05 18:06
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0064_shelfimage_shelfid'),
    ]

    operations = [
        migrations.AddField(
            model_name='shelfimage',
            name='image_name',
            field=models.CharField(default='', max_length=200),
        ),
    ]
