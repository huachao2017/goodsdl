# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-09-21 16:30
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('arm', '0002_armimage_tabel_z'),
    ]

    operations = [
        migrations.RenameField(
            model_name='armimage',
            old_name='tabel_z',
            new_name='table_z',
        ),
    ]