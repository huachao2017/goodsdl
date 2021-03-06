# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-12-13 10:52
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0015_auto_20171208_1451'),
    ]

    operations = [
        migrations.CreateModel(
            name='GoodsClass',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('class_type', models.IntegerField(default=0)),
                ('score', models.FloatField(default=0.0)),
                ('upc', models.CharField(max_length=20)),
                ('image_class', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='image_class_goods', to='goods.ImageClass')),
            ],
        ),
    ]
