from rest_framework.test import APIClient
from django.test import TestCase

from goods2.models import TaskLog
from PIL import Image
import tempfile


class TaskLogTestCase(TestCase):
    def setUp(self):
        pass

    def test_tasklog_create(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        task_log_qs = TaskLog.objects.filter(state=1)
        self.assertEqual(len(task_log_qs), 1)

    def test_tasklog_get(self):
        client = APIClient()
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        response = client.get('/api2/tasklog/')
        print(response)
        self.assertEqual(response.status_code, 200)
        task_log_list = response.data['results']
        self.assertEqual(len(task_log_list), 1)
        self.assertEqual(task_log_list[0]['name'], 'test')

    def test_image_create(self):
        image = Image.new('RGB', (100, 100))

        tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
        image.save(tmp_file)
        response = self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': tmp_file}, format='multipart')
        self.assertEqual(response.status_code, 201)
