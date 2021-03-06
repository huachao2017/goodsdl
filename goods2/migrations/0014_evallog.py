# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-06-19 19:11
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0013_auto_20180619_1817'),
    ]

    operations = [
        migrations.CreateModel(
            name='EvalLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('precision', models.FloatField(default=0.0)),
                ('checkpoint_step', models.IntegerField(default=0)),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('train_action', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='train_evals', to='goods2.TrainAction')),
            ],
        ),
    ]
