# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-09-20 17:43
from __future__ import unicode_literals

import arm.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ArmImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rgb_source', models.ImageField(max_length=200, upload_to=arm.models.image_upload_source)),
                ('depth_source', models.ImageField(max_length=200, upload_to=arm.models.image_upload_source)),
                ('result', models.TextField()),
                ('create_time', models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='date created')),
            ],
        ),
    ]
