from django.test import TestCase
from django.test import Client
from goods2.models import TaskLog

class ImageTestCase(TestCase):
    def setUp(self):
        pass

    def test_tasklog_create(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        task_log_qs = TaskLog.objects.filter(state=1)
        self.assertEqual(len(task_log_qs), 1)

    def test_image_get(self):
        c = Client()
        response = c.get('/api2/image')
        self.assertEqual(response.status_code,200)
