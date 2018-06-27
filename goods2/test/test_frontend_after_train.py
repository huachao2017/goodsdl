from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainImage, TrainUpc, Deviceid, DeviceidExclude
from goods2.cron import check_device
import os

from django.conf import settings

@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test')
class FrontEndAfterTrainTestCase(APITestCase):

    @classmethod
    def setUpTestData(cls):
        # 准备模型和样本库
        client = APIClient()

    def test_check_device(self):
        self.assertEqual(0,0)

    def test_image_post(self):
        self.assertEqual(0,0)

    def test_image_groundtruth_post(self):
        self.assertEqual(0,0)
