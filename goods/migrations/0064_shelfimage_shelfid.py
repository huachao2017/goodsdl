# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2019-04-29 18:37
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0063_auto_20190429_1712'),
    ]

    operations = [
        migrations.AddField(
            model_name='shelfimage',
            name='shelfid',
            field=models.IntegerField(db_index=True, default=0),
        ),
    ]
