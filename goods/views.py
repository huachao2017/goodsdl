import datetime
import logging
import os
import shutil
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image as im
from django.conf import settings
from object_detection.utils import visualization_utils as vis_util
from rest_framework import mixins
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from dl import imagedetectionV3, imagedetectionV4, imageclassifyV1, imagedetection_only_step1, imagedetection_only_step2, imagedetection_only_step3
# from dl.old import imagedetection
from dl.only_step2 import create_goods_tf_record
from dl.step1 import create_onegoods_tf_record, export_inference_graph as e1
from dl.step20 import convert_goods_step20
from dl.step30 import convert_goods_step30
from dl.step2 import convert_goods
from dl.step3 import convert_goods_step3
from dl import common
from .serializers import *
import json

logger = logging.getLogger("django")

class Test(APIView):
    def get(self, request):
        print(request.query_params)
        subprocess.call('nohup python3 /home/src/goodsdl/dl/step2/test.py  > /root/test.out 2>&1 &', shell=True)
        import sys
        path = sys.path
        return Response({'Test': path})


class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class ImageTestViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        # logger.info('begin create:')
        # 兼容没有那么字段的请求
        if 'lastinterval' not in request.data:
            request.data['lastinterval'] = 0.0

        logger.info('begin detect:{}'.format(request.data['deviceid']))
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        # 暂时性分解Detect，需要一个处理type编码
        if serializer.instance.deviceid == 'os1':
            export1s = ExportAction.objects.filter(train_action__action='T1').filter(checkpoint_prefix__gt=0).order_by('-update_time')[:1]

            if len(export1s)>0:
                detector = imagedetection_only_step1.ImageDetectorFactory_os1.get_static_detector(export1s[0].pk)
                step1_min_score_thresh = .5
                ret, aiinterval = detector.detect(serializer.instance, step1_min_score_thresh=step1_min_score_thresh)
                return Response(ret, status=status.HTTP_201_CREATED, headers=headers)
        elif serializer.instance.deviceid == 'os2':
            export2s = ExportAction.objects.filter(train_action__action='T2').filter(
                checkpoint_prefix__gt=0).order_by('-update_time')[:1]

            if len(export2s) > 0:
                detector = imagedetection_only_step2.ImageDetectorFactory_os2.get_static_detector(
                    export2s[0].pk,export2s[0].model_name)
                ret, aiinterval = detector.detect(serializer.instance)
            return Response(ret, status=status.HTTP_201_CREATED, headers=headers)
        elif serializer.instance.deviceid == 'os20':
            export20s = ExportAction.objects.filter(train_action__action='T20').filter(
                checkpoint_prefix__gt=0).order_by('-update_time')[:1]

            if len(export20s) > 0:
                detector = imagedetection_only_step2.ImageDetectorFactory_os2.get_static_detector(
                    export20s[0].pk, export20s[0].model_name)
                ret, aiinterval = detector.detect(serializer.instance)
            return Response(ret, status=status.HTTP_201_CREATED, headers=headers)
        elif serializer.instance.deviceid == 'os3':
            export2s = ExportAction.objects.filter(train_action__action='T2').filter(
                checkpoint_prefix__gt=0).order_by('-update_time')[:1]
            export3s = ExportAction.objects.filter(train_action__action='T3').filter(checkpoint_prefix__gt=0).order_by(
                '-update_time')

            if len(export2s) > 0:
                detector = imagedetection_only_step3.ImageDetectorFactory_os3.get_static_detector(
                    export2s[0].pk, export3s, export2s[0].model_name)
                ret, aiinterval = detector.detect(serializer.instance)
            return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        # logger.info('begin create:')
        # 兼容没有那么字段的请求
        if 'lastinterval' not in request.data:
            request.data['lastinterval'] = 0.0

        if request.data['deviceid'] in ['nnn', ]:
            logger.info('{} forbidden'.format(request.data['deviceid']))
            return Response([], status=status.HTTP_403_FORBIDDEN)
        logger.info('begin detect:{}'.format(request.data['deviceid']))
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        if serializer.instance.deviceid == 'nnn':
            pass
        else:
            last_images = Image.objects.filter(deviceid=serializer.instance.deviceid).filter(pk__lt=serializer.instance.pk).order_by('-create_time')[:5]

            last_image = None
            for one_image in last_images:
                if one_image.ret != '':
                    last_image = one_image
                    break

            from tracking.tracking import compare_same
            if last_image is not None and compare_same(serializer.instance.source.path, last_image.source.path):
                tmp_dir = os.path.join(os.path.split(serializer.instance.source.path)[0], 'tmp')
                if not tf.gfile.Exists(tmp_dir):
                    tf.gfile.MakeDirs(tmp_dir)

                # 移动检测重复图片
                shutil.move(serializer.instance.source.path, tmp_dir)
                Image.objects.get(pk=serializer.instance.pk).delete()
                ret_reborn = json.loads(last_image.ret)
                logger.info('duplicate detect:{},{}'.format(serializer.instance.deviceid, str(len(ret_reborn) if ret_reborn is not None else 0)))

                # 检测重复直接返回
                return Response(ret_reborn, status=status.HTTP_201_CREATED, headers=headers)

            aiinterval = .0
            # 正式应用区
            export1s = ExportAction.objects.filter(train_action__action='T1').filter(checkpoint_prefix__gt=0).order_by(
                '-update_time')[:1]

            if serializer.instance.deviceid == '275': # 测试step20
                export20s = ExportAction.objects.filter(train_action__action='T20').filter(train_action__serial='').filter(
                    checkpoint_prefix__gt=0).order_by('-update_time')[:1]

                if len(export20s) == 0:
                    logger.error('not found detection model!')
                    return Response([], status=status.HTTP_201_CREATED, headers=headers)
                detector = imagedetectionV4.ImageDetectorFactory.get_static_detector(export1s[0].pk, export20s[0].pk,
                                                                                     step2_model_name=export20s[0].model_name)
                step1_min_score_thresh = .9
                step2_min_score_thresh = .6

                ret, aiinterval = detector.detect(serializer.instance, step1_min_score_thresh=step1_min_score_thresh,
                                                  step2_min_score_thresh=step2_min_score_thresh)  # , compress=True)

            else:
                export2s = ExportAction.objects.filter(train_action__action='T2').filter(train_action__serial='').filter(checkpoint_prefix__gt=0).order_by(
                    '-update_time')[:1]
                export3s = ExportAction.objects.filter(train_action__action='T3').filter(train_action__serial='').filter(checkpoint_prefix__gt=0).order_by(
                    '-update_time')

                if len(export1s) == 0 or len(export2s) == 0:
                    logger.error('not found detection model!')
                    return Response([], status=status.HTTP_201_CREATED, headers=headers)

                detector = imagedetectionV3.ImageDetectorFactory.get_static_detector(export1s[0].pk, export2s[0].pk, export3s, step2_model_name = export2s[0].model_name)
                step1_min_score_thresh = .9
                step2_min_score_thresh = .6

                ret, aiinterval = detector.detect(serializer.instance, step1_min_score_thresh=step1_min_score_thresh,
                                      step2_min_score_thresh=step2_min_score_thresh) #, compress=True)

            if ret is None:
                logger.error('detection model is not ready')
                Image.objects.get(pk=serializer.instance.pk).delete()
                return Response([], status=status.HTTP_201_CREATED, headers=headers)

            ret_reborn = []
            index = 0
            class_index_dict = {}
            for goods in ret:
                # 兼容上一个版本
                if 'action' not in goods:
                    goods['action'] = 0
                Goods.objects.create(image_id=serializer.instance.pk,
                                     class_type=goods['class'],
                                     score1=goods['score'],
                                     score2=goods['score2'],
                                     upc=goods['upc'],
                                     xmin=goods['xmin'],
                                     ymin=goods['ymin'],
                                     xmax=goods['xmax'],
                                     ymax=goods['ymax'],
                                     )
                if goods['class'] in class_index_dict:
                    ret_reborn[class_index_dict[goods['class']]]['box'].append({
                        'score': goods['score'],
                        'score2': goods['score2'],
                        'action': goods['action'],
                        'xmin': goods['xmin'],
                        'ymin': goods['ymin'],
                        'xmax': goods['xmax'],
                        'ymax': goods['ymax'],
                    })
                else:
                    box = []
                    box.append({
                        'score': goods['score'],
                        'score2': goods['score2'],
                        'action': goods['action'],
                        'xmin': goods['xmin'],
                        'ymin': goods['ymin'],
                        'xmax': goods['xmax'],
                        'ymax': goods['ymax'],
                    })
                    ret_reborn.append({
                        'class': goods['class'],
                        'upc': goods['upc'],
                        'box': box
                    })
                    class_index_dict[goods['class']] = index
                    index = index + 1

            # 保存ai本次返回和计算时间
            serializer.instance.aiinterval = aiinterval
            serializer.instance.ret = json.dumps(ret_reborn, cls=NumpyEncoder)
            serializer.instance.save()
            logger.info('end detect:{},{}'.format(serializer.instance.deviceid, str(len(ret_reborn) if ret_reborn is not None else 0)))
            # logger.info('end create')
            # return Response({'Test':True})
            return Response(ret_reborn, status=status.HTTP_201_CREATED, headers=headers)

class ImageReportViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = ImageReport.objects.order_by('-id')
    serializer_class = ImageReportSerializer

class ImageClassViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                        viewsets.GenericViewSet):
    queryset = ImageClass.objects.order_by('-id')
    serializer_class = ImageClassSerializer

    def create(self, request, *args, **kwargs):
        # logger.info('begin create:')
        # 兼容没有那么字段的请求
        if 'deviceid' not in request.data:
            request.data['deviceid'] = get_client_ip(request)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        # 暂时性分解Detect，需要一个处理type编码
        detector = imageclassifyV1.ImageClassifyFactory.get_static_detector('1')
        min_score_thresh = .8
        logger.info('begin classify:{},{}'.format(serializer.instance.deviceid, serializer.instance.source.path))
        ret = detector.detect(serializer.instance.source.path, min_score_thresh=min_score_thresh)

        if ret is None or len(ret) <= 0:
            logger.info('end classify:0')
            # 删除无用图片
            os.remove(serializer.instance.source.path)
            ImageClass.objects.get(pk=serializer.instance.pk).delete()
        else:
            for goods_class in ret:
                GoodsClass.objects.create(image_class_id=serializer.instance.pk,
                                          class_type=goods_class['class'],
                                          score=goods_class['score'],
                                          upc=goods_class['upc'],
                                          )
            logger.info('end classify:{},{}'.format(serializer.instance.deviceid, str(len(ret))))

        # logger.info('end create')
        # return Response({'Test':True})
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

class ProblemGoodsViewSet(DefaultMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    queryset = ProblemGoods.objects.order_by('-id')
    serializer_class = ProblemGoodsSerializer


def get_client_ip(request):
    try:
        real_ip = request.META['HTTP_X_FORWARDED_FOR']
        regip = real_ip.split(",")[0]
    except:
        try:
            regip = request.META['REMOTE_ADDR']
        except:
            regip = ""
    return regip

def wrap_ret(ret):
    standard_ret = {
        'status': 200,
        'message': '成功',
        'attachment': ret,
    }
    return standard_ret

class GetSampleCount(APIView):
    def get(self, request):
        upc = request.query_params['upc']

        ret = {}

        dataset_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME)
        dirlist = os.listdir(dataset_dir)
        total_count = 0
        for dirname in dirlist:
            if dirname.startswith('data_new_'):
                deviceid = dirname.split('_')[-1]
                upc_dir = os.path.join(dataset_dir,dirname,upc)
                command = 'ls {} | grep -v visual | grep -v .xml | wc -l'.format(upc_dir)
                # logger.info(command)
                count = int(os.popen(command).readline())
                if count > 0:
                    ret[deviceid] = count
                    total_count += count
            elif dirname == 'data_new':
                ret['official'] = count
                total_count += count

        ret['total'] = total_count

        return Response(wrap_ret(ret), status=status.HTTP_200_OK)


class TrainImageViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainImage.objects.order_by('-id')
    serializer_class = TrainImageSerializer

    def create(self, request, *args, **kwargs):
        # 兼容没有那么字段的请求
        # if 'name' not in request.data :
        #     request.data['name'] = request.data['upc']
        if 'deviceid' not in request.data:
            request.data['deviceid'] = get_client_ip(request)
        if 'traintype' not in request.data:
            request.data['traintype'] = 0

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create image xml:{}'.format(serializer.instance.source.path))

        image_path = serializer.instance.source.path
        image = im.open(image_path)
        xml_path, _ = os.path.split(os.path.realpath(__file__))
        xml_path = os.path.join(xml_path, 'template.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        root.find('size').find('width').text = str(image.size[0])
        root.find('size').find('height').text = str(image.size[1])
        root.find('object').find('name').text = str(serializer.instance.upc)
        for box in root.iter('bndbox'):
            # 改变xml中的坐标值
            box.find('xmin').text = str(serializer.instance.xmin)
            box.find('ymin').text = str(serializer.instance.ymin)
            box.find('xmax').text = str(serializer.instance.xmax)
            box.find('ymax').text = str(serializer.instance.ymax)
        # 写入新的xml
        a, b = os.path.splitext(serializer.instance.source.path)
        tree.write(a + ".xml")

        # 画带box的图片
        image_dir = os.path.dirname(image_path)
        output_image_path = os.path.join(image_dir, 'visual_' + os.path.split(image_path)[-1])
        image = im.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        image_pil = im.fromarray(np.uint8(image_np)).convert('RGB')

        vis_util.draw_bounding_box_on_image(image_pil,
                                            serializer.instance.ymin, serializer.instance.xmin,
                                            serializer.instance.ymax, serializer.instance.xmax,
                                            color='DarkOrange',
                                            display_str_list=((serializer.instance.name if serializer.instance.name != '' else 'None'),),
                                            use_normalized_coordinates=False, thickness=2)

        np.copyto(image_np, np.array(image_pil))
        output_image = im.fromarray(image_np)
        output_image.thumbnail((int(im_width * 0.5), int(im_height * 0.5)), im.ANTIALIAS)
        output_image.save(output_image_path)

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        a, b = os.path.splitext(instance.source.path)
        dir = os.path.dirname(instance.source.path)
        if os.path.isfile(instance.source.path):
            os.remove(instance.source.path)
        if os.path.isfile(a + ".xml"):
            os.remove(a + ".xml")

        havefile = False
        for i in os.listdir(dir):
            havefile = True
            break
        if not havefile:
            shutil.rmtree(dir)

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

class TrainImageOnlyViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainImageOnly.objects.order_by('-id')
    serializer_class = TrainImageOnlySerializer

    def create(self, request, *args, **kwargs):
        # 兼容没有那么字段的请求
        # if 'name' not in request.data :
        #     request.data['name'] = request.data['upc']
        if 'deviceid' not in request.data:
            request.data['deviceid'] = get_client_ip(request)
        if 'traintype' not in request.data:
            request.data['traintype'] = 2

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

class TrainImageClassViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainImageClass.objects.order_by('-id')
    serializer_class = TrainImageClassSerializer

    def create(self, request, *args, **kwargs):
        # 兼容没有那么字段的请求
        # if 'name' not in request.data :
        #     request.data['name'] = request.data['upc']
        if 'traintype' not in request.data:
            request.data['traintype'] = 1

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        export1s = ExportAction.objects.filter(train_action__action='T1').filter(checkpoint_prefix__gt=0).order_by(
            '-update_time')[:1]

        if len(export1s) > 0:
            detector = imagedetection_only_step1.ImageDetectorFactory_os1.get_static_detector(export1s[0].pk)
            step1_min_score_thresh = .5
            ret, _ = detector.detect(serializer.instance, step1_min_score_thresh=step1_min_score_thresh)
            # to data_new

            if len(ret) >= 1:
                ymin = ret[0]['ymin']
                xmin = ret[0]['xmin']
                ymax = ret[0]['ymax']
                xmax = ret[0]['xmax']
                logger.info('create image xml:{}'.format(serializer.instance.source.path))

                image_path = serializer.instance.source.path
                image = im.open(image_path)
                xml_path, _ = os.path.split(os.path.realpath(__file__))
                xml_path = os.path.join(xml_path, 'template.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                root.find('size').find('width').text = str(image.size[0])
                root.find('size').find('height').text = str(image.size[1])
                root.find('object').find('name').text = str(serializer.instance.upc)
                for box in root.iter('bndbox'):
                    # 改变xml中的坐标值
                    box.find('xmin').text = str(xmin)
                    box.find('ymin').text = str(ymin)
                    box.find('xmax').text = str(xmax)
                    box.find('ymax').text = str(ymax)
                # 写入新的xml
                a, b = os.path.splitext(serializer.instance.source.path)
                tree.write(a + ".xml")

                # # 生成step2图片
                # upc_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, 'step2', serializer.instance.upc)
                # if not tf.gfile.Exists(upc_dir):
                #     tf.gfile.MakeDirs(upc_dir)
                # newimage = image.crop((xmin, ymin, xmax, ymax))
                # newimage_split = os.path.split(image_path)
                # new_image_path = os.path.join(upc_dir, newimage_split[1])
                # newimage.save(new_image_path, 'JPEG')

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class TrainActionViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainAction.objects.order_by('-id')
    serializer_class = TrainActionSerializer

    def create(self, request, *args, **kwargs):
        # 兼容没有字段的请求
        # if 'dataset_dir' not in request.data:
        #     request.data['dataset_dir'] = ''
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('create action:{}'.format(serializer.instance.action))

        if serializer.instance.action == 'T1':
            # 杀死原来的train
            # os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
            # os.system('ps -ef | grep eval.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')

            # 训练准备
            additional_data_dir = None
            if serializer.instance.traintype == 0:
                data_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, 'data')
                additional_data_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, 'data_raw')
            else:
                data_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, str(serializer.instance.traintype))

            fine_tune_checkpoint_dir = settings.TRAIN_ROOT
            if serializer.instance.is_fineture:
                export1s = ExportAction.objects.filter(train_action__action='T1').filter(
                    checkpoint_prefix__gt=0).order_by(
                    '-update_time')[:1]
                if len(export1s)>0:
                    fine_tune_checkpoint_dir = os.path.join(settings.TRAIN_ROOT, str(export1s[0].pk))

            label_map_dict = create_onegoods_tf_record.prepare_train(data_dir, settings.TRAIN_ROOT,
                                                                     str(serializer.instance.pk),
                                                                     fine_tune_checkpoint_dir,
                                                                     serializer.instance.is_fineture,
                                                                     additional_data_dir=additional_data_dir)

            train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))

            # 训练
            # --num_clones=2 同步修改config中的batch_size=2
            command = 'nohup python3 {}/step1/train.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --train_dir={} --local_fineture={}  > /root/train1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
                serializer.instance.is_fineture,
            )
            logger.info(command)
            subprocess.call(command, shell=True)
            # 评估
            command = 'nohup python3 {}/step1/eval.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --checkpoint_dir={} --eval_dir={}  > /root/eval1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
                os.path.join(train_logs_dir, 'eval_log'),
            )
            logger.info(command)
            subprocess.call(command, shell=True)
        elif serializer.instance.action == 'T20':
            self.train_step20(serializer.instance)
        elif serializer.instance.action == 'T21':
            self.train_step21(serializer.instance)
        elif serializer.instance.action == 'T30':
            self.train_step30(serializer.instance)
        elif serializer.instance.action == 'T2':
            self.train_step2(serializer.instance)
        elif serializer.instance.action == 'T3':
            self.train_step3(serializer.instance)
        elif serializer.instance.action == 'TC':
            self.train_only_step2(serializer.instance)

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def train_only_step2(self, actionlog):
        # 训练准备
        if actionlog.traintype == 0:
            data_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, 'data')
        else:
            data_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, str(actionlog.traintype))
        class_names_to_ids, training_filenames, validation_filenames = create_goods_tf_record.prepare_train(
            data_dir, settings.TRAIN_ROOT, str(actionlog.pk))
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(actionlog.pk))
        model_name = 'inception_resnet_v2'
        # 训练
        command = 'nohup python3 {}/only_step2/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --batch_size={} --learning_rate={} > /root/train_only2.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            len(training_filenames),
            model_name,
            32,
            0.01
        )
        logger.info(command)
        subprocess.call(command, shell=True)
        # 评估
        command = 'nohup python3 {}/only_step2/eval.py --dataset_split_name=validation --dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={}  > /root/eval_only2.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            os.path.join(train_logs_dir, 'eval_log'),
            len(validation_filenames),
            model_name
        )
        logger.info(command)
        subprocess.call(command, shell=True)

    def train_step20(self, actionlog):
        # TODO 需要实现fineture
        # 训练准备
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(actionlog.pk))
        source_dataset_dir = os.path.join(
            settings.MEDIA_ROOT,
            settings.DATASET_DIR_NAME,
            common.STEP20_PREFIX if actionlog.serial=='' else common.STEP20_PREFIX+'_'+str(actionlog.serial))

        class_names_to_ids, training_filenames, validation_filenames = convert_goods_step20.prepare_train(source_dataset_dir,
            train_logs_dir)
        # step2_model_name = 'inception_resnet_v2'
        # step2_model_name = 'nasnet_large'
        if actionlog.model_name == 'nasnet_large':
            batch_size = 8
        else:
            batch_size = 64

        train_steps = int(len(training_filenames) * 100 / batch_size) # 设定最大训练次数，每个样本进入网络100次，测试验证200次出现过拟合
        if train_steps < 20000:
            train_steps = 20000 # 小样本需要增加训练次数
        # 训练
        command = 'nohup python3 {}/step20/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --max_number_of_steps={} > /root/train20.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            len(training_filenames),
            actionlog.model_name,
            1,
            batch_size,
            train_steps
        )
        logger.info(command)
        subprocess.call(command, shell=True)
        # 评估
        command = 'nohup python3 {}/step20/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={} > /root/eval20.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            source_dataset_dir,
            train_logs_dir,
            os.path.join(train_logs_dir, 'eval_log'),
            len(validation_filenames),
            actionlog.model_name,
        )
        logger.info(command)
        subprocess.call(command, shell=True)

    def train_step21(self, actionlog):
        # TODO 需要实现fineture
        # 训练准备
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(actionlog.pk))
        source_dataset_dir = actionlog.dataset_dir
        class_names_to_ids, training_filenames, validation_filenames = convert_goods_step20.prepare_train(source_dataset_dir,
            train_logs_dir)
        # step2_model_name = 'inception_resnet_v2'
        # step2_model_name = 'nasnet_large'
        if actionlog.model_name == 'nasnet_large':
            batch_size = 8
        else:
            batch_size = 64

        train_steps = int(len(training_filenames) * 100 / batch_size) # 设定最大训练次数，每个样本进入网络100次，测试验证200次出现过拟合
        if train_steps < 20000:
            train_steps = 20000 # 小样本需要增加训练次数
        # 训练
        command = 'nohup python3 {}/step21/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --max_number_of_steps={} > /root/train21-{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            len(training_filenames),
            actionlog.model_name,
            1,
            batch_size,
            train_steps,
            actionlog.traintype
        )
        logger.info(command)
        subprocess.call(command, shell=True)
        # 评估
        command = 'nohup python3 {}/step21/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={} > /root/eval21-{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            source_dataset_dir,
            train_logs_dir,
            os.path.join(train_logs_dir, 'eval_log'),
            len(validation_filenames),
            actionlog.model_name,
            actionlog.traintype
        )
        logger.info(command)
        subprocess.call(command, shell=True)

    def train_step30(self, actionlog):
        # TODO 需要实现fineture
        # 训练准备
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(actionlog.pk))
        source_dataset_dir = actionlog.dataset_dir

        class_names_to_ids, training_filenames, validation_filenames = convert_goods_step30.prepare_train(source_dataset_dir,
            train_logs_dir)

        logger.info('train step30:{}'.format(source_dataset_dir))
        logger.info(class_names_to_ids)
        logger.info('sample count:{},{}'.format(len(training_filenames),len(validation_filenames)))
        # 更新actionlog.model_name
        if len(training_filenames) > 1000:
            actionlog.model_name = 'nasnet_large'
        else:
            actionlog.model_name = 'nasnet_mobile'
        actionlog.save()

        if actionlog.model_name == 'nasnet_large':
            batch_size = 8
        else:
            batch_size = 64

        train_steps = int(len(training_filenames) * 100 / batch_size) # 设定最大训练次数，每个样本进入网络100次，测试验证200次出现过拟合
        if actionlog.model_name == 'nasnet_large':
            if train_steps < 80000:
                train_steps = 80000 # 小样本需要增加训练次数
        else:
            if train_steps < 30000:
                train_steps = 30000 # 小样本需要增加训练次数


        # 更新TrainTask
        tasks = TrainTask.objects.filter(state=0).filter(dataset_dir=source_dataset_dir).order_by('-cluster_cnt')[:1]
        task = tasks[0]
        task.train_id = actionlog.pk
        task.category_cnt = len(class_names_to_ids)
        task.sample_cnt = len(training_filenames)
        task.step_cnt = train_steps
        task.model_name = actionlog.model_name
        task.save()

        # 训练
        command = 'nohup python3 {}/step30/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --max_number_of_steps={} > /root/train30-{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            len(training_filenames),
            actionlog.model_name,
            1,
            batch_size,
            train_steps,
            task.pk,
        )
        logger.info(command)
        subprocess.call(command, shell=True)
        # 评估
        command = 'nohup python3 {}/step30/eval2.py --dataset_split_name=validation --train_task_id={} --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={} > /root/eval30-{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            task.pk,
            train_logs_dir,
            source_dataset_dir,
            train_logs_dir,
            os.path.join(train_logs_dir, 'eval_log'),
            len(validation_filenames),
            actionlog.model_name,
            task.pk,
        )
        logger.info(command)
        subprocess.call(command, shell=True)

    def train_step2(self, actionlog):
        if actionlog.is_fineture:
            self.train_step2_with_fineture(actionlog)
            return

        # 训练准备
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(actionlog.pk))
        source_dataset_dir = os.path.join(
            settings.MEDIA_ROOT,
            settings.DATASET_DIR_NAME,
            common.STEP2_PREFIX if actionlog.serial=='' else common.STEP2_PREFIX+'_'+str(actionlog.serial))

        class_names_to_ids, training_filenames, validation_filenames = convert_goods.prepare_train(source_dataset_dir,
            train_logs_dir)
        # step2_model_name = 'inception_resnet_v2'
        # step2_model_name = 'nasnet_large'
        if actionlog.model_name == 'nasnet_large':
            batch_size = 8
        else:
            batch_size = 64

        train_steps = int(len(training_filenames) * 100 / batch_size) # 设定最大训练次数，每个样本进入网络100次，测试验证200次出现过拟合
        if train_steps < 20000:
            train_steps = 20000 # 小样本需要增加训练次数
        # 训练
        command = 'nohup python3 {}/step2/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --max_number_of_steps={} > /root/train2.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            len(training_filenames),
            actionlog.model_name,
            1,
            batch_size,
            train_steps
        )
        logger.info(command)
        subprocess.call(command, shell=True)
        # 评估
        command = 'nohup python3 {}/step2/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={} > /root/eval2.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            source_dataset_dir,
            train_logs_dir,
            os.path.join(train_logs_dir, 'eval_log'),
            len(validation_filenames),
            actionlog.model_name,
        )
        logger.info(command)
        subprocess.call(command, shell=True)

    def train_step2_with_fineture(self, actionlog):
        # 训练准备
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(actionlog.pk))
        source_dataset_dir = os.path.join(
            settings.MEDIA_ROOT,
            settings.DATASET_DIR_NAME,
            common.STEP2_PREFIX if actionlog.serial=='' else common.STEP2_PREFIX+'_'+str(actionlog.serial))

        # 获取step2最后导出的网络
        export2s = ExportAction.objects.filter(train_action__action='T2').filter(
            checkpoint_prefix__gt=0).order_by('-update_time')[:1]
        checkpoint_path = os.path.join(settings.BASE_DIR, 'dl/model', str(export2s[0].pk))
        fineture_label_path = os.path.join(checkpoint_path, 'labels.txt')

        class_names_to_ids, training_filenames, validation_filenames = convert_goods.prepare_train_with_fineture(source_dataset_dir,
            train_logs_dir, fineture_label_path)
        # step2_model_name = 'inception_resnet_v2'
        # step2_model_name = 'nasnet_large'
        batch_size = 8
        # 训练
        command = 'nohup python3 {}/step2/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --max_number_of_steps={} ' \
                  '--checkpoint_path={} --checkpoint_exclude_scopes=final_layer,aux_11/aux_logits/FC --trainable_scopes=final_layer,aux_11/aux_logits/FC > /root/train2.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            len(training_filenames),
            actionlog.model_name,
            1,
            batch_size,
            int(len(training_filenames) * 100 / batch_size),  # 设定最大训练次数，每个样本进入网络100次，目前不清楚funiture的过拟合情况
            checkpoint_path,
        )
        logger.info(command)
        subprocess.call(command, shell=True)
        # 评估
        command = 'nohup python3 {}/step2/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={} ' \
                  ' > /root/eval2.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            source_dataset_dir,
            train_logs_dir,
            os.path.join(train_logs_dir, 'eval_log'),
            len(validation_filenames),
            actionlog.model_name,
        )
        logger.info(command)
        subprocess.call(command, shell=True)

    def train_step3(self, actionlog):
        # 训练准备
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(actionlog.pk))
        source_dataset_dir = os.path.join(
            settings.MEDIA_ROOT,
            settings.DATASET_DIR_NAME,
            common.STEP3_PREFIX if actionlog.serial == '' else common.STEP3_PREFIX + '_' + str(actionlog.serial),
            str(actionlog.traintype))
        class_names_to_ids, training_filenames, validation_filenames = convert_goods_step3.prepare_train(source_dataset_dir,
            train_logs_dir)

        if class_names_to_ids is None:
            logger.error('class_names_to_ids is None!')
            return
        # step3_model_name = 'inception_resnet_v2'
        # step3_model_name = 'nasnet_mobile'
        batch_size = 64 if actionlog.model_name=='nasnet_mobile' else 8
        # 训练
        command = 'nohup python3 {}/step3/train.py --dataset_split_name=train --dataset_dir={} --train_dir={} --example_num={} --model_name={} --num_clones={} --batch_size={} --max_number_of_steps={}  > /root/train3-{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            train_logs_dir,
            len(training_filenames),
            actionlog.model_name,
            1,
            batch_size,
            int(len(training_filenames) * 10000 / batch_size),  # 设定最大训练次数，每个样本进入网络10000次
            str(actionlog.traintype)
        )
        logger.info(command)
        subprocess.call(command, shell=True)
        # 评估
        command = 'nohup python3 {}/step3/eval2.py --dataset_split_name=validation --dataset_dir={} --source_dataset_dir={} --checkpoint_path={} --eval_dir={} --example_num={} --model_name={} > /root/eval3-{}.out 2>&1 &'.format(
            os.path.join(settings.BASE_DIR, 'dl'),
            train_logs_dir,
            source_dataset_dir,
            train_logs_dir,
            os.path.join(train_logs_dir, 'eval_log'),
            len(validation_filenames),
            actionlog.model_name,
            str(actionlog.traintype)
        )
        logger.info(command)
        subprocess.call(command, shell=True)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if serializer.instance.action == 'T1':
            # 杀死原来的train
            # os.system('ps -ef | grep train.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')
            # os.system('ps -ef | grep eval.py | grep -v grep | cut -c 9-15 | xargs kill -s 9')

            if serializer.instance.traintype == 2:
                # TODO 这是一个临时功能
                data_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, 'data_raw')
                create_onegoods_tf_record.prepare_rawdata_update_train(data_dir, settings.TRAIN_ROOT, str(serializer.instance.pk))
            train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.pk))
            # 继续训练 --num_clones=2
            command = 'nohup python3 {}/step1/train.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --train_dir={}  > /root/train1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
            )
            logger.info(command)
            subprocess.call(command, shell=True)
            # 评估
            command = 'nohup python3 {}/step1/eval.py --pipeline_config_path={}/faster_rcnn_nas_goods.config --checkpoint_dir={} --eval_dir={}  > /root/eval1.out 2>&1 &'.format(
                os.path.join(settings.BASE_DIR, 'dl'),
                train_logs_dir,
                train_logs_dir,
                os.path.join(train_logs_dir, 'eval_log'),
            )
            logger.info(command)
            subprocess.call(command, shell=True)

        elif serializer.instance.action == 'T20':
            self.train_step20(serializer.instance)
        elif serializer.instance.action == 'T30':
            self.train_step30(serializer.instance)
        elif serializer.instance.action == 'T2':
            self.train_step2(serializer.instance)
        elif serializer.instance.action == 'T3':
            self.train_step3(serializer.instance)
        elif serializer.instance.action == 'TC':
            self.train_only_step2(serializer.instance)
        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(instance.pk))
        if os.path.isdir(train_logs_dir):
            shutil.rmtree(train_logs_dir)

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

class ExportActionViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = ExportAction.objects.order_by('-id')
    serializer_class = ExportActionSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('export train:{}'.format(serializer.instance.train_action.pk))
        if serializer.instance.model_name == '':
            serializer.instance.model_name = serializer.instance.train_action.model_name
            serializer.instance.save()

        prefix = None
        if serializer.instance.train_action.action == 'T1':
            prefix = self.export_detection_graph(serializer.instance.train_action, serializer)

        elif serializer.instance.train_action.action in ['T2','T3','T20','T30']:
            prefix = self.export_classify_graph(serializer.instance.train_action, serializer)

        elif serializer.instance.train_action.action == 'TC':
            prefix = self.export_classify_graph(serializer.instance.train_action, serializer)

        if prefix is None:
            ExportAction.objects.get(pk=serializer.instance.pk).delete()
        else:
            serializer.instance.checkpoint_prefix = int(prefix)
            serializer.instance.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def export_detection_graph(self, train_action, serializer):
        prefix = None
        logger.info('Export <trainid:{}> Graph from detection train.'.format(train_action.pk))
        trained_checkpoint_dir = os.path.join(settings.TRAIN_ROOT, str(train_action.pk))
        checkpoint_model_path = tf.train.latest_checkpoint(trained_checkpoint_dir)
        if checkpoint_model_path:
            prefix = checkpoint_model_path.split('-')[-1]
            model_dir = os.path.join(settings.BASE_DIR, 'dl', 'model', str(serializer.instance.pk))
            if not tf.gfile.Exists(model_dir):
                tf.gfile.MakeDirs(model_dir)
            # 输出pb
            e1.export(os.path.join(trained_checkpoint_dir, 'faster_rcnn_nas_goods.config'),
                      checkpoint_model_path,
                      model_dir,
                      )

            export_file_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
            label_file_path = os.path.join(model_dir, 'goods_label_map.pbtxt')
            # copy label
            shutil.copy(os.path.join(settings.TRAIN_ROOT, str(train_action.pk), 'goods_label_map.pbtxt'),
                        label_file_path)
        return prefix

    def export_classify_graph(self, train_action, serializer):
        prefix = None
        logger.info('Export <trainid:{}> Graph from classify train.'.format(train_action.pk))
        trained_checkpoint_dir = os.path.join(settings.TRAIN_ROOT, str(train_action.pk))
        checkpoint_model_path = tf.train.latest_checkpoint(trained_checkpoint_dir)
        if checkpoint_model_path:
            prefix = checkpoint_model_path.split('-')[-1]
            model_dir = os.path.join(settings.BASE_DIR, 'dl', 'model', str(serializer.instance.pk))
            if not tf.gfile.Exists(model_dir):
                tf.gfile.MakeDirs(model_dir)
            checkpoint_file_path = os.path.join(model_dir, 'checkpoint')
            # 输出pb
            # e2.export(step2_model_name, trained_checkpoint_dir, export_file_path)
            # 重写checkpoint file
            with open(checkpoint_file_path, 'w') as output:
                a = os.path.split(checkpoint_model_path)
                output.write('model_checkpoint_path: "{}"\n'.format(os.path.join(model_dir, a[1])))
                output.write('all_model_checkpoint_paths: "{}"\n'.format(os.path.join(model_dir, a[1])))
            shutil.copy(checkpoint_model_path + '.data-00000-of-00001', model_dir)
            shutil.copy(checkpoint_model_path + '.index', model_dir)
            shutil.copy(checkpoint_model_path + '.meta', model_dir)
            # copy dataset
            # shutil.copy(os.path.join(settings.TRAIN_ROOT, str(train_action.pk), 'goods_recogonize_train.tfrecord'),
            #             model_dir)
            # copy label
            shutil.copy(os.path.join(settings.TRAIN_ROOT, str(train_action.pk), 'labels.txt'), model_dir)

            if train_action.action == 'T2':
                # copy cluster
                shutil.copy(os.path.join(settings.TRAIN_ROOT, str(train_action.pk), common.CLUSTER_FILE_NAME), model_dir)
            # reboot django
            # os.utime(os.path.join(settings.BASE_DIR, 'main', 'settings.py'), None)
        return prefix

class StopTrainActionViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                             viewsets.GenericViewSet):
    queryset = StopTrainAction.objects.order_by('-id')
    serializer_class = StopTrainActionSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        logger.info('stop train:{}'.format(serializer.instance.train_action.pk))

        train_logs_dir = os.path.join(settings.TRAIN_ROOT, str(serializer.instance.train_action.pk))
        train_ps = os.popen('ps -ef | grep train.py | grep {} | grep -v grep'.format(train_logs_dir)).readline()
        if train_ps != '':
            pid = int(train_ps.split()[1])
            os.system('kill -s 9 {}'.format(str(pid)))
        eval_ps = os.popen('ps -ef | grep eval | grep {} | grep -v grep'.format(train_logs_dir)).readline()
        if eval_ps != '':
            pid = int(eval_ps.split()[1])
            os.system('kill -s 9 {}'.format(str(pid)))

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class RfidImageCompareActionViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin,
                                    mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = RfidImageCompareAction.objects.order_by('-id')
    serializer_class = RfidImageCompareActionSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        domain = 'admin.fastxbox.cn'
        startTime = int(time.mktime(serializer.instance.startTime.timetuple()) * 1000)
        endTime = int(time.mktime(serializer.instance.endTime.timetuple()) * 1000)
        deviceid = serializer.instance.deviceid

        logger.info(
            'begin compare:{}:{},{}:{}'.format(serializer.instance.startTime, startTime, serializer.instance.endTime,
                                               endTime))

        response = urllib.request.urlopen(
            'http://{}/payment/order-by-ai.json?deviceId={}&startTime={}&endTime={}'.format(domain, deviceid, startTime,
                                                                                            endTime))
        html = response.read()
        rfid_ret = json.loads(html)
        print(rfid_ret)
        if rfid_ret['status'] == 200:
            for transaction in rfid_ret['attachment']:
                transaction_time = transaction['payTime']
                # 时间转换
                transaction_time = datetime.datetime.fromtimestamp(transaction_time / 1e3)
                # 查询image
                images = Image.objects.filter(deviceid=deviceid).filter(create_time__lt=transaction_time).order_by(
                    '-id')[:1]
                if len(images) > 0:
                    image_id = images[0].pk
                else:
                    image_id = None
                rfid_transaction = RfidTransaction.objects.create(image_id=image_id,
                                                                  transaction_time=transaction_time,
                                                                  )
                image_upc_to_count = {}
                if len(images) > 0:
                    for image_goods in images[0].image_goods:
                        if image_goods.upc in image_upc_to_count:
                            image_upc_to_count[image_goods.upc] = image_upc_to_count[image_goods.upc] + 1
                        else:
                            image_upc_to_count[image_goods.upc] = 1

                same_upc_num = 0
                only_rfid_upc_num = 0
                only_image_upc_num = 0
                for one_rfid_upc in transaction['upcModels']:
                    RfidGoods.objects.create(rfid_transaction_id=rfid_transaction.pk,
                                             upc=one_rfid_upc['upc'],
                                             count=one_rfid_upc['count']
                                             )
                    # caculate
                    if one_rfid_upc['upc'] in image_upc_to_count:
                        if one_rfid_upc['count'] >= image_upc_to_count[one_rfid_upc['upc']]:
                            same_upc_num = same_upc_num + image_upc_to_count[one_rfid_upc['upc']]
                            only_rfid_upc_num = only_rfid_upc_num + one_rfid_upc['count'] - image_upc_to_count[
                                one_rfid_upc['upc']]
                        else:
                            same_upc_num = same_upc_num + one_rfid_upc['count']
                            only_image_upc_num = only_image_upc_num + image_upc_to_count[one_rfid_upc['upc']] - \
                                                 one_rfid_upc['count']
                    else:
                        only_rfid_upc_num = only_rfid_upc_num + one_rfid_upc['count']

                TransactionMetrix.objects.create(rfid_transaction_id=rfid_transaction.pk,
                                                 same_upc_num=same_upc_num,
                                                 only_rfid_upc_num=only_rfid_upc_num,
                                                 only_image_upc_num=only_image_upc_num,
                                                 )

        logger.info('response:{}'.format(html))

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class TransactionMetrixViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = TransactionMetrix.objects.order_by('-id')
    serializer_class = TransactionMetrixSerializer

class TrainTaskViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainTask.objects.order_by('-update_time')
    serializer_class = TrainTaskSerializer

class ClusterStructureViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = ClusterStructure.objects.order_by('-update_time')
    serializer_class = ClusterStructureSerializer

class ClusterEvalDataViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = ClusterEvalData.objects.order_by('-id')
    serializer_class = ClusterEvalDataSerializer

class ClusterSampleScoreViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = ClusterSampleScore.objects.order_by('-id')
    serializer_class = ClusterSampleScoreSerializer

class ClusterUpcScoreViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = ClusterUpcScore.objects.order_by('-id')
    serializer_class = ClusterUpcScoreSerializer
