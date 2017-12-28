# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-12-28 16:48
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0034_timelog'),
    ]

    operations = [
        migrations.CreateModel(
            name='PreStep2TimeLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('param', models.CharField(max_length=500)),
                ('total', models.FloatField(default=0.0)),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='image_prestep2timelog', to='goods.Image')),
            ],
        ),
    ]
