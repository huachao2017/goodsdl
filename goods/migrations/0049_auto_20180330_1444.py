# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-03-30 14:44
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0048_trainaction_serial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainaction',
            name='action',
            field=models.CharField(choices=[('T1', 'Train Step 1'), ('T20', 'Train Step 2_0'), ('T2', 'Train Step 2'), ('T3', 'Train Step 3'), ('TC', 'Train Only Step 2')], max_length=5),
        ),
    ]
