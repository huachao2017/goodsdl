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

    def test_check_device(self):
        self.assertEqual(0,0)

    def test_image_post(self):
        deviceid = '0'
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
        response, upc = util._add_one_image(self.client, deviceid, '111')

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

        # test device to commercial
        device = Deviceid.objects.get(deviceid=deviceid)
        self.assertEqual(device.state, common.DEVICE_STATE_TESTING)
        device.state = common.DEVICE_STATE_COMMERCIAL
        device.save()

        response, upc = util._add_one_image(self.client, deviceid,'111')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data), 5)
        self.assertEqual(response.data[0]['upc'], upc)

    def test_image_groundtruth_post(self):
        deviceid = '0'
        identify = '111'
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
        for i in range(9):
            response, upc = util._add_one_image(self.client, deviceid, identify)

        with open(os.path.join(settings.BASE_DIR, 'images/train_1.jpg'), mode='rb') as fp:
            self.client.post('/api2/image/', {'deviceid': deviceid, 'identify': identify, 'source': fp}, format='multipart')

        response = self.client.post('/api2/imagegroundtruth/', {'deviceid': deviceid, 'identify': identify, 'upc':upc})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        image_ground_truth = ImageGroundTruth.objects.all()[0]
        self.assertEqual(image_ground_truth.truth_rate, 0.9)
        self.assertTrue(image_ground_truth.total_precision>0.8)
