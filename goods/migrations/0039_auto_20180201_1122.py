# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-02-01 11:22
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0038_exportaction_model_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='exportaction',
            name='model_name',
            field=models.CharField(choices=[('ANY', 'not specify'), ('T2_INV2', 'inception resnat V2'), ('T2_NASL', 'nas large')], default='ANY', max_length=10),
        ),
    ]
