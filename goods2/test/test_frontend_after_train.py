from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainAction, TrainModel, ImageTrainModel, TrainImage, TrainUpc, Deviceid, DeviceidExclude
from goods2.cron import check_device
import os
import datetime
from goods2 import common
from goods2.test import util

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
        train_action = TrainAction.objects.create(
            action='TA',
            ip='192.168.1.170',
            state=common.TRAIN_STATE_COMPLETE
        )
        train_model = TrainModel.objects.create(
            train_action_id=train_action.pk,
            checkpoint_step=243856,
            precision=0.999,
            model_path='/home/src/goodsdl/dl/model/75'
        )
        response, upc = util._add_one_image(self.client, '0','111')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        image_train_model_qs = ImageTrainModel.objects.all()
        self.assertEqual(len(image_train_model_qs), 1)
        self.assertEqual(image_train_model_qs[0].train_model.pk, train_model.pk)

        image_qs = Image.objects.all()
        self.assertEqual(len(image_qs), 1)
        image = image_qs[0]
        image_result_qs = image.image_results.order_by('-score')
        self.assertEqual(len(image_result_qs), 5)
        first_image_result = image_result_qs[0]
        self.assertEqual(first_image_result.upc, upc)

    def test_image_groundtruth_post(self):
        self.assertEqual(0,0)
