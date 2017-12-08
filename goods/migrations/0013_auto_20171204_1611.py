# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-12-04 16:11
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('goods', '0012_auto_20171201_1340'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProblemGoods',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('index', models.IntegerField(default=0)),
                ('class_type_0', models.IntegerField(default=0)),
                ('class_type_1', models.IntegerField(default=0)),
                ('class_type_2', models.IntegerField(default=0)),
                ('class_type_3', models.IntegerField(default=0)),
                ('class_type_4', models.IntegerField(default=0)),
                ('score_0', models.FloatField(default=0.0)),
                ('score_1', models.FloatField(default=0.0)),
                ('score_2', models.FloatField(default=0.0)),
                ('score_3', models.FloatField(default=0.0)),
                ('score_4', models.FloatField(default=0.0)),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='image_problem_goods', to='goods.Image')),
            ],
        ),
        migrations.RemoveField(
            model_name='goods',
            name='image',
        ),
        migrations.DeleteModel(
            name='Goods',
        ),
    ]