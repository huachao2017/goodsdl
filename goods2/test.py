from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from goods2.models import TaskLog, Image, TrainImage, TrainUpc
import os

from django.conf import settings

class FrontEndTestCase(APITestCase):
    def setUp(self):
        pass

    def test_tasklog_create(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        task_log_qs = TaskLog.objects.filter(state=1)
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

    def test_image_groundtruth_post(self):
        with open(os.path.join(settings.BASE_DIR, 'images/test_1.jpg'), mode='rb') as fp:
            self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')
        with open(os.path.join(settings.BASE_DIR, 'images/train_1.jpg'), mode='rb') as fp:
            self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')

        image_qs = Image.objects.filter(identify='1111')
        self.assertEqual(len(image_qs), 2)
        response = self.client.post('/api2/imagegroundtruth/', {'identify': '1111', 'upc': '111'})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

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
        train_upc_qs = TrainUpc.objects.filter(upc='111')
        self.assertEqual(len(train_upc_qs), 1)
        self.assertEqual(train_upc_qs[0].cnt, 1)

class CronTestCase(APITestCase):

    @classmethod
    def setUpTestData(cls):
        # 上传2类图片各10张
        client = APIClient()

    def test_transfer_sample(self):
        self.assertEqual(0,0)

    def test_create_train_TA(self):
        self.assertEqual(0,0)

    def test_create_train_TF(self):
        self.assertEqual(0,0)

    def test_create_train_TC(self):
        self.assertEqual(0,0)

    def test_execute_train(self):
        self.assertEqual(0,0)

    def test_check_train(self):
        self.assertEqual(0,0)