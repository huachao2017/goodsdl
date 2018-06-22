from django.test import TestCase
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from goods2.models import TaskLog
from PIL import Image
import tempfile
from io import BytesIO

class TaskLogTestCase(APITestCase):
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
        # tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
        # tmp_file.name='a.jpg'
        # image.save(tmp_file)

        image = Image.new('RGB', (100, 100))

        img = BytesIO(image.tobytes())
        img.name='myimage2.jpg'
        response = self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': img}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
