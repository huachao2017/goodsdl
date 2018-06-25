from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainImage, TrainUpc, Deviceid, DeviceidExclude
from goods2.cron import check_device, transfer_sample, create_train, execute_train, check_train
import os

from django.conf import settings

@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test')
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

        image = Image.objects.filter(identify='1111')[0]
        self.assertEqual(image.deviceid, '0')
        self.assertTrue(os.path.isfile(image.source.path))
        device = Deviceid.objects.get(deviceid=image.deviceid)
        self.assertEqual(device.state, 0)

    def test_image_groundtruth_post(self):
        with open(os.path.join(settings.BASE_DIR, 'images/test_1.jpg'), mode='rb') as fp:
            self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')
        with open(os.path.join(settings.BASE_DIR, 'images/train_1.jpg'), mode='rb') as fp:
            self.client.post('/api2/image/', {'deviceid': '0', 'identify': '1111', 'source': fp}, format='multipart')

        image_qs = Image.objects.filter(identify='1111')
        self.assertEqual(len(image_qs), 2)
        response = self.client.post('/api2/imagegroundtruth/', {'deviceid': '0', 'identify': '1111', 'upc': '111'})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        response = self.client.post('/api2/imagegroundtruth/', {'deviceid': '0', 'identify': '1111', 'upc': '111'})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        image_ground_truth_qs = ImageGroundTruth.objects.filter(deviceid='0').filter(identify='1111')
        self.assertEqual(image_ground_truth_qs[0].cnt, 2)
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
        train_upc_qs = TrainUpc.objects.filter(upc='111')
        self.assertEqual(len(train_upc_qs), 1)
        self.assertEqual(train_upc_qs[0].cnt, 1)

    def test_deviceid_exclude_post(self):
        response = self.client.post('/api2/deviceexclude/',{'deviceid': '500'})
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        device_exclude = DeviceidExclude.objects.get(deviceid='500')
        self.assertEqual(device_exclude.deviceid, '500')


@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test')
class CronBeforeTrainTestCase(APITestCase):

    @classmethod
    def setUpTestData(cls):
        client = APIClient()
        # 排除deviceid=500
        client.post('/api2/deviceexclude/', {'deviceid': '500'})

        # 上传2类图片各10张
        dataset_root_path=os.path.join(settings.MEDIA_ROOT, 'dataset','step2')
        upcs = ['4711931005106', '4714221811227']

        for upc in upcs:
            upc_path = os.path.join(dataset_root_path, upc)
            index = 0
            for filename in os.listdir(upc_path):
                image_path = os.path.join(upc_path, filename)
                with open(image_path, mode='rb') as fp:
                    response = client.post('/api2/trainimage/', {'deviceid': '1000', 'upc':upc, 'source': fp}, format='multipart')

                index += 1
                if index >= 10:
                    break

    def _add_image(self, deviceid, identify):
        dataset_root_path=os.path.join(settings.MEDIA_ROOT, 'dataset','step2')
        upcs = ['4711931005106', '4714221811227']

        for upc in upcs:
            upc_path = os.path.join(dataset_root_path, upc)
            index = 0
            for filename in os.listdir(upc_path):
                image_path = os.path.join(upc_path, filename)
                with open(image_path, mode='rb') as fp:
                    response = self.client.post('/api2/image/', {'deviceid': deviceid, 'identify': upc+identify, 'source': fp}, format='multipart')

                index += 1
                if index >= 10:
                    break

            self.client.post('/api2/imagegroundtruth/', {'deviceid': deviceid, 'identify': upc+identify, 'upc': upc})

    def test_transfer_sample(self):
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),20)
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,10)

        self._add_image('1000', '0')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),20)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),1)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),22) # 增加两个样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,11)

        self._add_image('500', '1')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),40)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),2)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),22) # 不增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,11)

        self._add_image('1000', '2')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),60)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),3)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),24) # 再增加2样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,12)


    def test_create_train_TA(self):
        self.assertEqual(0,0)

    def test_create_train_TF(self):
        self.assertEqual(0,0)

    def test_create_train_TC(self):
        self.assertEqual(0,0)

@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test')
class CronBetweenTrainTestCase(APITestCase):

    @classmethod
    def setUpTestData(cls):
        # 上传2类图片各10张
        client = APIClient()

    def test_execute_train(self):
        self.assertEqual(0,0)

    def test_check_train(self):
        self.assertEqual(0,0)

@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test')
class CronAfterTrainTestCase(APITestCase):

    @classmethod
    def setUpTestData(cls):
        # 上传2类图片各10张
        client = APIClient()

    def test_check_device(self):
        self.assertEqual(0,0)

@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test')
class FrontEndAfterTrainTestCase(APITestCase):

    @classmethod
    def setUpTestData(cls):
        # 准备模型和样本库
        client = APIClient()

    def test_image_post(self):
        self.assertEqual(0,0)

    def test_image_groundtruth_post(self):
        self.assertEqual(0,0)
