from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainImage, Deviceid, DeviceidTrain
from goods2 import common
import os

from django.conf import settings

@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test')
class FrontEndTestCase(APITestCase):
    def setUp(self):
        pass

    def test_tasklog_create(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        task_log_qs = TaskLog.objects.filter(state=common.TASK_STATE_DOING)
        self.assertEqual(len(task_log_qs), 1)

    def test_tasklog_get(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        response = self.client.get('/api2/tasklog/')
        # print(response.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 1)
        task_log_list = response.data['results']
        self.assertEqual(len(task_log_list), 1)
        self.assertEqual(task_log_list[0]['name'], 'test')

    def test_image_post(self):
        with open(os.path.join(settings.BASE_DIR, 'images/test_1.jpg'), mode='rb') as fp:
            response = self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        image = Image.objects.filter(identify='1111')[0]
        self.assertEqual(image.deviceid, '0')
        self.assertTrue(os.path.isfile(image.source.path))
        device = Deviceid.objects.get(deviceid=image.deviceid)
        self.assertEqual(device.state, common.DEVICE_STATE_TESTING)

    def test_image_groundtruth_post(self):
        with open(os.path.join(settings.BASE_DIR, 'images/test_1.jpg'), mode='rb') as fp:
            self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')
        with open(os.path.join(settings.BASE_DIR, 'images/train_1.jpg'), mode='rb') as fp:
            self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')

        image_qs = Image.objects.filter(identify='1111')
        self.assertEqual(len(image_qs), 2)
        response = self.client.post('/api2/imagegroundtruth/', {'deviceid': '0', 'identify': '1111', 'upc': '111'})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        image_ground_truth_qs = ImageGroundTruth.objects.filter(deviceid='0').filter(identify='1111')
        self.assertEqual(image_ground_truth_qs[0].cnt, 2)
        self.assertEqual(image_ground_truth_qs[0].upc, '111')
        self.assertEqual(image_ground_truth_qs[0].truth_rate, 0.0)
        self.assertEqual(image_ground_truth_qs[0].precision, 0.0)

        response = self.client.post('/api2/imagegroundtruth/', {'deviceid': '0', 'identify': '1111', 'upc': '222'})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        image_ground_truth_qs = ImageGroundTruth.objects.filter(deviceid='0').filter(identify='1111')
        self.assertEqual(image_ground_truth_qs[0].cnt, 2)
        self.assertEqual(image_ground_truth_qs[0].upc, '222')
        self.assertEqual(image_ground_truth_qs[0].truth_rate, 0.0)
        self.assertEqual(image_ground_truth_qs[0].precision, 0.0)

        image_qs = Image.objects.filter(identify='1111')
        last_image = image_qs[0]
        image_ground_truth = last_image.image_ground_truth
        self.assertEqual(image_ground_truth.upc, '111')

    def test_train_image_post(self):
        with open(os.path.join(settings.BASE_DIR, 'images/test_1.jpg'), mode='rb') as fp:
            response = self.client.post('/api2/trainimage/', {'deviceid': '0', 'upc':'111', 'source': fp}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        train_image_qs = TrainImage.objects.filter(upc='111')
        train_image = train_image_qs[0]
        self.assertEqual(train_image.source_from, 1)

    def test_deviceid_train_post(self):
        response = self.client.post('/api2/devicetrain/',{'deviceid': '500'})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        device_exclude = DeviceidTrain.objects.get(deviceid='500')
        self.assertEqual(device_exclude.deviceid, '500')

