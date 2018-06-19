# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-06-15 19:32
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0008_auto_20180614_1424'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainimage',
            name='source_image',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='trainimages', to='goods2.Image'),
        ),
        migrations.AlterField(
            model_name='image',
            name='image_ground_truth',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='ground_truth_images', to='goods2.ImageGroundTruth'),
        ),
        migrations.AlterField(
            model_name='trainaction',
            name='f_model',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='child_trains', to='goods2.TrainModel'),
        ),
    ]
