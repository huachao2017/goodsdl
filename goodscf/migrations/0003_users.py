# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-07-13 15:12
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goodscf', '0002_usergoods'),
    ]

    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('openid', models.CharField(default='', max_length=50, unique=True)),
            ],
        ),
    ]
