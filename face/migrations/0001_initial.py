# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-08-07 12:56
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('picurl', models.CharField(max_length=200)),
                ('index', models.TextField()),
                ('create_time', models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='date created')),
            ],
        ),
    ]
