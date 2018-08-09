import logging
import shutil
import os

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from goods2.dl import imagedetection
from goods2 import common

from goods2.serializers import *

logger = logging.getLogger("django2")


class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

class DeviceidViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = Deviceid.objects.order_by('-id')
    serializer_class = DeviceidSerializer

class DeviceidExcludeViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = DeviceidExclude.objects.order_by('-id')
    serializer_class = DeviceidExcludeSerializer


class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        logger.info('begin detect image:{},{}'.format(request.data['deviceid'], request.data['identify']))

        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            headers = self.get_success_headers(serializer.data)
        except Exception as e:
            logger.error(e)
            raise e

        try:
            device = Deviceid.objects.get(deviceid=serializer.instance.deviceid)
        except Deviceid.DoesNotExist as e:
            device = Deviceid.objects.create(
                deviceid=serializer.instance.deviceid,
            )

        # 检测阶段
        last_normal_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).filter(deviceid=serializer.instance.deviceid).exclude(action='TC').order_by('-id')
        if len(last_normal_train_qs)>0:
            last_train = last_normal_train_qs[0]
            last_normal_train_model = TrainModel.objects.filter(train_action_id=last_train.pk).exclude(model_path='').order_by('-id')[0]
            detector = imagedetection.ImageDetectorFactory.get_static_detector(
                last_normal_train_model)
            upcs, scores = detector.detect(serializer.instance)
            ImageTrainModel.objects.create(
                train_model_id=last_normal_train_model.pk,
                image_id=serializer.instance.pk
            )

            last_tc_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).filter(deviceid=serializer.instance.deviceid).filter(action='TC').filter(complete_time__gt=last_normal_train_model.create_time).order_by('-id')
            if len(last_tc_train_qs)>0:
                last_tc_train = last_tc_train_qs[0]
                last_tc_train_model = TrainModel.objects.filter(train_action_id=last_tc_train.pk).exclude(model_path='').order_by('-id')[0]
                detector2 = imagedetection.ImageDetectorFactory.get_static_detector(
                    last_tc_train_model)
                upcs2, scores2 = detector2.detect(serializer.instance)
                ImageTrainModel.objects.create(
                    train_model_id=last_tc_train_model.pk,
                    image_id=serializer.instance.pk
                )
                # 联合计算
                upc_to_scores = {}
                for i in range(len(upcs)):
                    if upcs[i] in upc_to_scores:
                        upc_to_scores[upcs[i]] = upc_to_scores[upcs[i]]*0.5 + scores[i]*0.5
                    else:
                        upc_to_scores[upcs[i]] = scores[i]
                for i in range(len(upcs2)):
                    if upcs2[i] in upc_to_scores:
                        upc_to_scores[upcs2[i]] = upc_to_scores[upcs2[i]]*0.5 + scores2[i]*0.5
                    else:
                        upc_to_scores[upcs2[i]] = scores2[i]

                upcs, scores = sort_upc_to_scores(upc_to_scores)

            # 输出结果
            for i in range(len(upcs)):
                if i < 5: # 不超过5个
                    ImageResult.objects.create(
                        image_id=serializer.instance.pk,
                        upc=upcs[i],
                        score=scores[i]
                    )

        ret = []
        if device.state >= common.DEVICE_STATE_COMMERCIAL:
            # 没有商用的不返回结果
            upc_to_scores = {}
            decay = 1
            image_qs = Image.objects.filter(identify=serializer.instance.identify).order_by('-id')
            for image in image_qs:
                image_result_qs = image.image_results.all()
                for image_result in image_result_qs:
                    upc = image_result.upc
                    score = image_result.score
                    if upc in upc_to_scores:
                        upc_to_scores[upc] += upc_to_scores[upc] + score*decay
                    else:
                        upc_to_scores[upc] = score*decay
                decay -= 0.1 #前面次数衰减
                if decay <= 0:
                    break

            upcs, scores = sort_upc_to_scores(upc_to_scores)
            for i in range(len(upcs)):
                if i < 5:  # 不超过5个
                    ret.append(
                        {
                            'upc': upcs[i],
                            'score': scores[i],
                        }
                    )
        elif device.deviceid == '36':
            ret = [
                {'upc': '2000000000103', 'score': 0.99},
                {'upc': '2000000000097', 'score': 0.99},
                {'upc': '2000000000093', 'score': 0.99},
                {'upc': '2000000000106', 'score': 0.99},
            ]
        elif device.deviceid == '3061':# 苹果
            ret = [
                {'upc': '2000000001540', 'score': 0.07},
                {'upc': '2000000001598', 'score': 0.25},
                {'upc': '3287583', 'score': 0.04},
                {'upc': '3313222', 'score': 0.01},
            ]
        elif device.deviceid == '3062':# 香蕉
            ret = [
                {'upc': '2000000001541', 'score': 0.95},
                {'upc': '3960271', 'score': 0.03},
                {'upc': '3283458', 'score': 0.01},
                {'upc': '3283452', 'score': 0.01},
            ]

        logger.info(ret)
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

def sort_upc_to_scores(upc_to_scores):
    items = upc_to_scores.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)

    upcs = [backitems[i][1] for i in range(0, len(backitems))]
    scores = [backitems[i][0] for i in range(0, len(backitems))]

    return upcs, scores


class ImageGroundTruthViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    queryset = ImageGroundTruth.objects.order_by('-id')
    serializer_class = ImageGroundTruthSerializer

    def create(self, request, *args, **kwargs):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            headers = self.get_success_headers(serializer.data)
        except Exception as e:
            if isinstance(e, ValidationError):
                try:
                    logger.info(request.data)
                    logger.info(serializer.data)
                    image_ground_truth = ImageGroundTruth.objects.get(identify=serializer.data['identify'])
                    image_ground_truth.upc = serializer.data['upc']
                    image_ground_truth.save()
                    serializer.instance = image_ground_truth
                    headers = self.get_success_headers(serializer.data)
                except Exception as e:
                    logger.error(e)
                    raise e
            else:
                logger.error(e)
                raise e

        images = Image.objects.filter(identify=serializer.instance.identify).filter(deviceid=serializer.instance.deviceid)
        truth_image_result_cnt = 0
        false_image_result_cnt = 0
        total_precision = 0.0
        for image in images:
            truth_image_result_qs = image.image_results.filter(upc=serializer.instance.upc)
            if len(truth_image_result_qs)>0:
                truth_image_result_cnt += 1
                total_precision += truth_image_result_qs[0].score
            else:
                false_image_result_cnt += 1
            image.image_ground_truth=serializer.instance
            image.save()

        serializer.instance.cnt = len(images)
        if truth_image_result_cnt+false_image_result_cnt > 0:
            serializer.instance.truth_rate = truth_image_result_cnt/(truth_image_result_cnt+false_image_result_cnt)
            serializer.instance.precision = total_precision/(truth_image_result_cnt+false_image_result_cnt)
        else:
            serializer.instance.truth_rate = 0.0
            serializer.instance.precision = 0.0
        serializer.instance.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class TrainImageViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainImage.objects.order_by('-id')
    serializer_class = TrainImageSerializer

    def create(self, request, *args, **kwargs):
        if 'source_from' not in request.data:
            request.data['source_from'] = 1
        if 'score' not in request.data:
            request.data['score'] = 1.0
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)


        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()

        os.remove(instance.source.path)
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)


class TrainActionViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    queryset = TrainAction.objects.order_by('-id')
    serializer_class = TrainActionSerializer


class TrainModelViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = TrainModel.objects.order_by('-id')
    serializer_class = TrainModelSerializer

class TaskLogViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = TaskLog.objects.order_by('-id')
    serializer_class = TaskLogSerializer
