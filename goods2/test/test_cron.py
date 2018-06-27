from rest_framework.test import APITestCase

from django.test import override_settings
from goods2.models import TaskLog, Image, ImageGroundTruth, TrainImage, TrainUpc, TrainAction, TrainModel, EvalLog
from goods2.cron import transfer_sample, create_train, execute_train, check_train, get_host_ip
import os
import shutil
import time
from goods2 import common
from goods2.test import util

from django.conf import settings


@override_settings(DETECT_DIR_NAME='images_test', DATASET_DIR_NAME='dataset_test',
                   TRAIN_ROOT=os.path.join(settings.BASE_DIR, 'train_test'),
                   MODEL_ROOT=os.path.join(settings.BASE_DIR, 'dl', 'model_test'))
class CronBeforeTrainTestCase(APITestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(common.get_dataset_dir(True), True)
        shutil.rmtree(common.get_train_path(), True)
        shutil.rmtree(common.get_model_path(), True)

    def test_transfer_sample(self):
        # 排除deviceid=500
        self.client.post('/api2/deviceexclude/', {'deviceid': '500'})
        util._add_train_image(self.client, upcs=['4711931005106', '4714221811227'])

        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs), 20)
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt, 10)

        util._add_image(self.client, '1000', '0')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()), 20)
        self.assertEqual(len(TaskLog.objects.filter(state=10)), 1)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs), 22)  # 增加两个样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt, 11)

        util._add_image(self.client, '500', '1')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()), 40)
        self.assertEqual(len(TaskLog.objects.filter(state=10)), 2)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs), 22)  # 不增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt, 11)

        util._add_image(self.client, '1000', '2')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()), 60)
        self.assertEqual(len(TaskLog.objects.filter(state=10)), 3)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs), 24)  # 再增加2样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt, 12)

        util._add_image(self.client, '1000', '3', add_ground_truth=False)
        transfer_sample()
        self.assertEqual(len(Image.objects.all()), 80)
        self.assertEqual(len(TaskLog.objects.filter(state=10)), 4)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs), 24)  # 不再增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt, 12)

        util._add_image(self.client, '1000', '4')
        transfer_sample()
        self.assertEqual(len(Image.objects.all()), 100)
        self.assertEqual(len(TaskLog.objects.filter(state=10)), 5)
        train_image_qs = TrainImage.objects.all()
        self.assertEqual(len(train_image_qs), 26)  # 不再增加样本
        train_upc = TrainUpc.objects.get(upc='4711931005106')
        self.assertEqual(train_upc.cnt, 13)

    def test_create_train_TA(self):
        create_train()
        self.assertEqual(len(TaskLog.objects.filter(state=10)), 1)
        self.assertEqual(len(TrainAction.objects.all()), 0)

        for i in range(100):
            util._add_train_image(self.client, upcs=['4711931005106', '4714221811227'])
        create_train()
        self.assertEqual(len(TaskLog.objects.filter(state=10)), 2)
        train_action = TrainAction.objects.filter(action='TA').filter(state=1)[0]
        self.assertEqual(train_action.train_cnt, 2000)
        self.assertEqual(train_action.validation_cnt, int(2000 * 0.3))
        train_action_upcs_qs = train_action.upcs.all()
        self.assertEqual(len(train_action_upcs_qs), 2)
        self.assertEqual(train_action_upcs_qs[0].cnt, 1000)

        my_ip = get_host_ip()
        if my_ip == '192.168.1.170':
            execute_train()
            self.assertEqual(len(TaskLog.objects.filter(state=10)), 3)
            train_action = TrainAction.objects.filter(action='TA').filter(state=5)[0]
            time.sleep(1)
            self.assertTrue(common.get_train_pid(train_action) > 0)
            self.assertTrue(common.get_eval_pid(train_action) > 0)
            common.stop_train_ps(train_action)

        train_action.state = 10
        train_action.save()
        train_model = TrainModel.objects.create(
            train_action_id=train_action.pk,
            checkpoint_step=100,
            precision=0.9,
            model_path='/test/a/'
        )
        create_train()
        self.assertEqual(len(TrainAction.objects.all()), 1)


    def test_create_train_TF_from_TA(self):
        for i in range(100):
            util._add_train_image(self.client, upcs=['4711931005106', '4714221811227'])
        create_train()

        train_action_ta = TrainAction.objects.filter(action='TA').filter(state=1)[0]
        train_action_ta.state = 10
        train_action_ta.save()

        train_model = TrainModel.objects.create(
            train_action_id=train_action_ta.pk,
            checkpoint_step=100,
            precision=0.9,
            model_path='/test/a/'
        )
        for i in range(10):
            util._add_train_image(self.client, upcs=['4711931005106'])
        create_train()
        train_action_tf_qs = TrainAction.objects.filter(action='TF').filter(state=1)
        self.assertEqual(len(train_action_tf_qs), 0)

        for i in range(10):
            util._add_train_image(self.client, upcs=['4711931005106'])

        self.assertEqual(len(TrainImage.objects.all()), 2200)
        create_train()
        train_action_tf = TrainAction.objects.filter(action='TF').filter(state=1)[0]
        self.assertEqual(train_action_tf.f_model.pk, train_model.pk)
        self.assertEqual(train_action_tf.train_cnt, 800)
        self.assertEqual(train_action_tf.validation_cnt, 600)
        train_action_tf_upcs_qs = train_action_tf.upcs.filter(upc='4711931005106')
        self.assertEqual(train_action_tf_upcs_qs[0].cnt, 1200)
        my_ip = get_host_ip()
        if my_ip == '192.168.1.170':
            execute_train()
            train_action = TrainAction.objects.filter(action='TF').filter(state=5)[0]
            time.sleep(1)
            self.assertTrue(common.get_train_pid(train_action) > 0)
            self.assertTrue(common.get_eval_pid(train_action) > 0)
            common.stop_train_ps(train_action)

    def test_create_train_TC_from_TA(self):
        for i in range(100):
            util._add_train_image(self.client, upcs=['4711931005106', '4714221811227'])
        create_train()

        train_action = TrainAction.objects.filter(action='TA').filter(state=1)[0]
        train_action.state = 10
        train_action.save()

        train_model = TrainModel.objects.create(
            train_action_id=train_action.pk,
            checkpoint_step=100,
            precision=0.9,
            model_path='/test/a/'
        )
        util._add_train_image(self.client, upcs=['6901668002525'])
        self.assertEqual(len(TrainImage.objects.all()), 2010)
        self.assertEqual(len(TrainUpc.objects.all()), 3)
        create_train()
        train_action_tc = TrainAction.objects.filter(action='TC').filter(state=1)[0]
        self.assertEqual(train_action_tc.f_model.pk, train_model.pk)
        self.assertEqual(train_action_tc.train_cnt, 30)
        self.assertEqual(train_action_tc.validation_cnt, 30)
        train_action_tc_upcs_qs = train_action_tc.upcs.filter(upc='6901668002525')
        self.assertEqual(train_action_tc_upcs_qs[0].cnt, 10)
        my_ip = get_host_ip()
        if my_ip == '192.168.1.173':
            execute_train()
            train_action = TrainAction.objects.filter(action='TC').filter(state=5)[0]
            time.sleep(1)
            self.assertTrue(common.get_train_pid(train_action) > 0)
            self.assertTrue(common.get_eval_pid(train_action) > 0)
            common.stop_train_ps(train_action)

    def test_check_train(self):
        my_ip = get_host_ip()
        if my_ip == '192.168.1.170':
            for i in range(100):
                util._add_train_image(self.client, upcs=['4711931005106', '4714221811227'])
            create_train()
            execute_train()
            self.assertEqual(len(TaskLog.objects.filter(state=10)), 2)
            train_action = TrainAction.objects.filter(action='TA').filter(state=5)[0]

            # 增加1次eval_log
            EvalLog.objects.create(
                train_action_id=train_action.pk,
                precision=0.95,
                checkpoint_step=100,
            )
            check_train()
            self.assertEqual(len(TaskLog.objects.filter(state=10)), 3)
            train_model_qs = TrainModel.objects.filter(train_action_id=train_action.pk)
            self.assertEqual(len(train_model_qs), 1)
            self.assertEqual(train_model_qs[0].model_path, '')

            # 增加10次eval_log
            for i in range(10):
                EvalLog.objects.create(
                    train_action_id=train_action.pk,
                    precision=0.998,
                    checkpoint_step=1000+i*10,
                )
            check_train()
            self.assertEqual(len(TaskLog.objects.filter(state=10)), 4)
            train_model_qs = TrainModel.objects.filter(train_action_id=train_action.pk)
            self.assertEqual(len(train_model_qs), 2)
            self.assertEqual(train_model_qs[0].model_path, os.path.join(common.get_model_path(), str(train_model_qs[0].pk)))
            train_action_qs = TrainAction.objects.filter(action='TA').filter(state=10)
            self.assertEqual(len(train_action_qs), 1)

            common.stop_train_ps(train_action)
        elif my_ip == '192.168.1.173':
            pass