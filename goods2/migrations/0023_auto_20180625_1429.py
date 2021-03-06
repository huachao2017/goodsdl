# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-06-25 14:29
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0022_deviceidprecision'),
    ]

    operations = [
        migrations.RenameField(
            model_name='deviceidprecision',
            old_name='deviceid',
            new_name='device',
        ),
        migrations.AddField(
            model_name='imagegroundtruth',
            name='cnt',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='imagegroundtruth',
            name='deviceid',
            field=models.CharField(db_index=True, default='0', max_length=20),
        ),
        migrations.AddField(
            model_name='imagegroundtruth',
            name='precision',
            field=models.FloatField(default=0.0),
        ),
    ]
