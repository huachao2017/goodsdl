import datetime
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import tensorflow as tf
from django.conf import settings
from rest_framework import mixins
from rest_framework import status
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
import goods.util

from dl import common
from dl import imagedetectionV3, imagedetectionV3_S, imageclassifyV1, imagedetection_only_step1, \
    imagedetection_only_step2, imagedetection_only_step3, imagedetection
# from dl.old import imagedetection
from .serializers import *

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
                ret, aiinterval,_ = detector.detect(serializer.instance.source.path, step1_min_score_thresh=step1_min_score_thresh)
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
                if one_image.ret != '' and os.path.isfile(one_image.source.path):
                    last_image = one_image
                    break

            from tradition.tracking.tracking import compare_same
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

            # if serializer.instance.deviceid == '275': # 10类的演示
            #     detector = imagedetection.ImageDetectorFactory.get_static_detector('10')
            #     min_score_thresh = .5
            #     logger.info('begin detect:{},{}'.format(serializer.instance.deviceid, serializer.instance.source.path))
            #     ret = detector.detect(serializer.instance.source.path, min_score_thresh=min_score_thresh)
            #
            step1_min_score_thresh = .9
            step2_min_score_thresh = .6
            if serializer.instance.deviceid == '': # 290': # 楼下演示
                detector = imagedetectionV3.ImageDetectorFactory.get_static_detector(serializer.instance.deviceid)
            else:# step1+step2+模式类的演示
                detector = imagedetectionV3_S.ImageDetectorFactory.get_static_detector(serializer.instance.deviceid)

            if detector is None:
                return Response([], status=status.HTTP_201_CREATED, headers=headers)

            ret, aiinterval = detector.detect(serializer.instance, step1_min_score_thresh=step1_min_score_thresh,
                                  step2_min_score_thresh=step2_min_score_thresh) #, compress=True)

            if ret is None:
                logger.error('detection model is not ready')
                Image.objects.get(pk=serializer.instance.pk).delete()
                return Response([], status=status.HTTP_201_CREATED, headers=headers)

            ret_reborn = []
            index = 0
            upc_index_dict = {}
            for goods in ret:
                # 兼容上一个版本
                if 'action' not in goods:
                    goods['action'] = 0
                if 'score2' not in goods:
                    goods['score2'] = 0
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
                if goods['upc'] in upc_index_dict:
                    ret_reborn[upc_index_dict[goods['upc']]]['box'].append({
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
                    upc_index_dict[goods['upc']] = index
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


class VerifyCnt(APIView):
    def get(self, request):
        deviceid = request.query_params['deviceid']
        paymentID = request.query_params['paymentID']
        picurl = request.query_params['picurl']
        goodscnt = int(request.query_params['goodscnt'])

        ret = {
            'paymentID':paymentID
        }
        export1s = ExportAction.objects.filter(train_action__action='T1').filter(checkpoint_prefix__gt=0).order_by(
            '-update_time')[:1]

        if len(export1s) > 0:
            detector = imagedetection_only_step1.ImageDetectorFactory_os1.get_static_detector(export1s[0].pk)
            step1_min_score_thresh = .8
            media_dir = settings.MEDIA_ROOT
            # 通过 picurl 获取图片
            now = datetime.datetime.now()
            image_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, deviceid, now.strftime('%Y%m%d'))
            if not tf.gfile.Exists(image_dir):
                tf.gfile.MakeDirs(image_dir)
            image_path = os.path.join(image_dir, '{}.jpg'.format(now.strftime('%H%M%S')))
            logger.info(image_path)
            urllib.request.urlretrieve(picurl, image_path)
            detect_ret, aiinterval, visual_image_path = detector.detect(image_path, step1_min_score_thresh=step1_min_score_thresh)
            ret['verifycnt'] = len(detect_ret)
            if len(detect_ret) > goodscnt:
                ret['isverify'] = 0
            else:
                ret['isverify'] = 1

            if visual_image_path is not None:
                visual_image_path = visual_image_path.replace(settings.MEDIA_ROOT+'/', settings.MEDIA_URL)
                ret['verifypicurl'] = visual_image_path
            else:
                ret['verifypicurl'] = ''

        return Response(goods.util.wrap_ret(ret), status=status.HTTP_200_OK)
