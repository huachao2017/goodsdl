# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-11-28 15:52
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0036_image_is_hand'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainimage',
            name='special_type',
            field=models.IntegerField(choices=[(0, 'NAN'), (1, 'hand'), (2, 'bag')], default=0),
        ),
    ]
