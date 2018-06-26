from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainImage, TrainUpc, Deviceid, DeviceidExclude, TrainAction
from goods2.cron import check_device, transfer_sample, create_train, execute_train, check_train
import os
import shutil
from goods2 import common

from django.conf import settings


def _add_train_image(client):
    # 上传2类图片各10张
    dataset_root_path = os.path.join(settings.MEDIA_ROOT, 'dataset', 'step2')
    upcs = ['4711931005106', '4714221811227']

    for upc in upcs:
        upc_path = os.path.join(dataset_root_path, upc)
        index = 0
        for filename in os.listdir(upc_path):
            image_path = os.path.join(upc_path, filename)
            with open(image_path, mode='rb') as fp:
                response = client.post('/api2/trainimage/', {'deviceid': '1000', 'upc': upc, 'source': fp},
                                       format='multipart')

            index += 1
            if index >= 10:
                break


def _add_image(client, deviceid, identify, add_ground_truth=True):
    dataset_root_path = os.path.join(settings.MEDIA_ROOT, 'dataset', 'step2')
    upcs = ['4711931005106', '4714221811227']

    for upc in upcs:
        upc_path = os.path.join(dataset_root_path, upc)
        index = 0
        for filename in os.listdir(upc_path):
            image_path = os.path.join(upc_path, filename)
            with open(image_path, mode='rb') as fp:
                response = client.post('/api2/image/',
                                            {'deviceid': deviceid, 'identify': upc + identify, 'source': fp},
                                            format='multipart')

            index += 1
            if index >= 10:
                break

        if add_ground_truth:
            client.post('/api2/imagegroundtruth/', {'deviceid': deviceid, 'identify': upc + identify, 'upc': upc})


@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test', TRAIN_ROOT=os.path.join(settings.BASE_DIR, 'train_test'))
class CronBeforeTrainTestCase(APITestCase):

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(common.get_dataset_dir(True), True)
        shutil.rmtree(common.get_train_path(), True)

    def test_transfer_sample(self):
        # 排除deviceid=500
        self.client.post('/api2/deviceexclude/', {'deviceid': '500'})
        _add_train_image(self.client)

        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),20)
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,10)

        _add_image(self.client, '1000', '0')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),20)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),1)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),22) # 增加两个样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,11)

        _add_image(self.client, '500', '1')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),40)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),2)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),22) # 不增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,11)

        _add_image(self.client, '1000', '2')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),60)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),3)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),24) # 再增加2样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,12)

        _add_image(self.client, '1000', '3', add_ground_truth=False)
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),80)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),4)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),24) # 不再增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,12)

        _add_image(self.client, '1000', '4')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),100)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),5)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),26) # 不再增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,13)


    def test_create_train_TA(self):
        create_train()
        self.assertEqual(len(TaskLog.objects.filter(state=10)),1)
        self.assertEqual(len(TrainAction.objects.all()),0)

        for i in range(101):
            _add_train_image(self.client)
        create_train()
        self.assertEqual(len(TaskLog.objects.filter(state=10)),2)
        train_action = TrainAction.objects.filter(action='TA').filter(state=1)[0]
        self.assertEqual(train_action.train_cnt, 2020)
        self.assertEqual(train_action.validation_cnt, int(2020*0.3))
        train_action_upcs_qs = train_action.upcs.all()
        self.assertEqual(len(train_action_upcs_qs), 2)
        self.assertEqual(train_action_upcs_qs[0].cnt, 1010)

    def test_create_train_TF(self):
        self.assertEqual(0,0)

    def test_create_train_TC(self):
        self.assertEqual(0,0)
