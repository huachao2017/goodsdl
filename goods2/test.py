from django.test import TestCase
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from goods2.models import TaskLog
from PIL import Image
from io import BytesIO
import os

from django.conf import settings

class DebugTestCase(APITestCase):
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

    def test_image_create(self):
        # image = Image.new('RGB', (100, 100))
        #
        # img = BytesIO(image.tobytes('jpeg'))
        # img.name='myimage2.jpg'
        with open(os.path.join(settings.BASE_DIR, 'images/test_1.jpg'), mode='rb') as fp:
            response = self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
