# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2019-01-26 15:37
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('track', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TransferRecognition',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('transfer_deviceid', models.CharField(db_index=True, max_length=20)),
                ('distance', models.IntegerField(default=0)),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
            ],
        ),
        migrations.AlterField(
            model_name='recognition',
            name='r_sid',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='transferrecognition',
            name='recognition',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='transfer_recognition', to='track.Recognition'),
        ),
    ]
