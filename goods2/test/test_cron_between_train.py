from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainImage, TrainUpc, Deviceid, DeviceidExclude
from goods2.cron import check_device, transfer_sample, create_train, execute_train, check_train
import os
import shutil
from goods2 import common

from django.conf import settings

@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test', TRAIN_ROOT=os.path.join(settings.BASE_DIR, 'train_test'))
class CronBetweenTrainTestCase(APITestCase):

    @classmethod
    def setUpTestData(cls):
        # 上传2类图片各10张
        client = APIClient()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(common.get_dataset_dir(True), True)
        shutil.rmtree(common.get_train_path(), True)

    def test_execute_train(self):
        self.assertEqual(0,0)

    def test_check_train(self):
        self.assertEqual(0,0)
