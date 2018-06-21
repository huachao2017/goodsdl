from django.test import TestCase
from goods2.models import TaskLog

class ImageTestCase(TestCase):
    def setUp(self):
        pass

    def test_image_create(self):
        task_log = TaskLog.objects.create(name='test', ip='test', message='')
        task_log_qs = TaskLog.objects.filter(state=1)
        self.assertEqual(len(task_log_qs), 1)