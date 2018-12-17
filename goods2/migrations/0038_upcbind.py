# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-12-17 16:25
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods2', '0037_trainimage_special_type'),
    ]

    operations = [
        migrations.CreateModel(
            name='UpcBind',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('deviceid', models.CharField(db_index=True, default='', max_length=20)),
                ('upc1', models.CharField(db_index=True, max_length=20)),
                ('upc2', models.CharField(db_index=True, max_length=20)),
                ('create_time', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('update_time', models.DateTimeField(auto_now=True, verbose_name='date updated')),
            ],
        ),
    ]
