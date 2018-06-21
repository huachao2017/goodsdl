import logging
import shutil

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from goods2.dl import imagedetection

from goods2.serializers import *

logger = logging.getLogger("django")


class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100


class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        # 检测阶段
        ret = []
        last_normal_train_qs = TrainAction.objects.filter(state=10).exclude(action='TC').order_by('-id')
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

            last_tc_train_qs = TrainAction.objects.filter(state=10).filter(action='TC').filter(update_time__gt=last_normal_train_model.create_time).order_by('-id')
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

                items = upc_to_scores.items()
                backitems = [[v[1], v[0]] for v in items]
                backitems.sort(reverse=True)

                upcs = [backitems[i][0] for i in range(0, len(backitems))]
                scores = [backitems[i][1] for i in range(0, len(backitems))]

            # 输出结果
            ret = []
            for i in range(len(upcs)):
                ret.append(
                    {
                        'upc': upcs[i],
                        'score': scores[i],
                    }
                )
                ImageResult.objects.create(
                    image_id=serializer.instance.pk,
                    upc=upcs[i],
                    score=scores[i]
                )

        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)


class ImageGroundTruthViewSet(DefaultMixin, mixins.CreateModelMixin,viewsets.GenericViewSet):
    queryset = ImageGroundTruth.objects.order_by('-id')
    serializer_class = ImageGroundTruthSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)

        images = Image.objects.filter(identify=serializer.instance.identify)
        for image in images:
            image.image_ground_truth=serializer.instance
            image.save()

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

        # add or update TrainUpc
        try:
            train_upc = TrainUpc.objects.get(upc=serializer.instance.upc)
            train_upc.cnt += 1
            train_upc.save()
        except TrainUpc.DoesNotExist as e:
            TrainUpc.objects.create(
                upc=serializer.instance.upc,
                cnt=1,
            )

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        train_upc = TrainUpc.objects.get(upc=instance.upc)
        train_upc.cnt -= 1
        train_upc.save()

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)


class TrainUpcViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = TrainUpc.objects.order_by('-id')
    serializer_class = TrainUpcSerializer

class TrainActionViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    queryset = TrainAction.objects.order_by('-id')
    serializer_class = TrainActionSerializer


class TrainModelViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainModel.objects.order_by('-id')
    serializer_class = TrainModelSerializer