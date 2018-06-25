from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainImage, TrainUpc, Deviceid, DeviceidExclude
from goods2.cron import check_device, transfer_sample, create_train, execute_train, check_train
import os

from django.conf import settings

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

    def _add_image(self, deviceid, identify, add_ground_truth=True):
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

            if add_ground_truth:
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

        self._add_image('1000', '3', add_ground_truth=False)
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),80)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),4)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),24) # 不再增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,12)

        self._add_image('1000', '4')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()),100)
        self.assertEqual(len(TaskLog.objects.filter(state=10)),5)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs),26) # 不再增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt,13)


    def test_create_train_TA(self):
        self.assertEqual(0,0)

    def test_create_train_TF(self):
        self.assertEqual(0,0)

    def test_create_train_TC(self):
        self.assertEqual(0,0)
