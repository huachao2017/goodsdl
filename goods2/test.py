from django.test import TestCase
from goods2.models import TaskLog
from io import BytesIO


class TaskLogTestCase(TestCase):
    reset_sequences = True

    def setUp(self):
        pass

    def test_tasklog_create(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        task_log_qs = TaskLog.objects.filter(state=1)
        self.assertEqual(len(task_log_qs), 1)

    def test_tasklog_get(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        response = self.client.get('/api2/tasklog/')
        self.assertEqual(response.status_code, 200)
        task_log_list = response.data['results']
        self.assertEqual(len(task_log_list), 1)
        self.assertEqual(task_log_list[0]['name'], 'test')

    def test_image_create(self):
        img = BytesIO(b'mybinarydata')
        img.name = 'myimage.jpg'
        response = self.client.post('api2/image/', {'deviceid': '0', 'identify': '1111', 'source': img})
        self.assertEqual(response.status_code, 201)
