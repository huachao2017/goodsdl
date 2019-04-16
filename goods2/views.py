import logging
import shutil
import os

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.decorators import action
from django_filters.rest_framework import DjangoFilterBackend
from goods2.dl import imagedetection
from goods2 import common
from goods2 import util
from tradition.hand.hand_detect import HandDetect

from goods2.serializers import *

logger = logging.getLogger("django2")


class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

class DeviceidViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = Deviceid.objects.order_by('-id')
    serializer_class = DeviceidSerializer

class DeviceidTrainViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = DeviceidTrain.objects.order_by('-id')
    serializer_class = DeviceidTrainSerializer


class UserImageViewSet(DefaultMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet, mixins.DestroyModelMixin):
    queryset = Image.objects.exclude(image_ground_truth_id=None).filter(is_train=False).filter(is_hand=False).order_by('-id')
    serializer_class = UserImageSerializer
    filter_backends = (DjangoFilterBackend,)
    filter_fields = ('deviceid', 'upc')

    @action(methods=['get'], detail=False)
    def device_list(self, request):
        devices = Image.objects.values('deviceid').distinct()
        ret = []
        for deviceid in devices:
            ret.append(deviceid['deviceid'])
        return Response(ret)

    @action(methods=['get'], detail=False)
    def upc_list(self, request):
        if 'deviceid' in request.query_params:
            deviceid = request.query_params['deviceid']
            upcs = Image.objects.exclude(image_ground_truth_id=None).filter(is_train=False).filter(is_hand=False).filter(deviceid=deviceid).values('upc').distinct()
        else:
            upcs = Image.objects.exclude(image_ground_truth_id=None).filter(is_train=False).filter(is_hand=False).values('upc').distinct()
        ret = []
        for upc in upcs:
            ret.append(upc['upc'])
        return Response(ret)

    @action(methods=['put'], detail=True)
    def add_to_train(self, request, pk=None):
        instance = self.get_object()
        # serializer = self.get_serializer(instance, data=request.data)
        # serializer.is_valid(raise_exception=True)

        train_source = '{}/{}/{}/{}'.format(common.get_dataset_dir(), instance.deviceid, instance.upc,
                                            'image_' + os.path.basename(instance.source.path))
        train_source_dir = '{}/{}/{}'.format(common.get_dataset_dir(True), instance.deviceid,
                                             instance.upc)
        import tensorflow as tf
        if not tf.gfile.Exists(train_source_dir):
            tf.gfile.MakeDirs(train_source_dir)
        train_source_path = '{}/{}'.format(train_source_dir, 'image_' + os.path.basename(instance.source.path))
        shutil.copy(instance.source.path, train_source_path)
        instance.is_train = True
        instance.save()
        TrainImage.objects.create(
            deviceid=instance.deviceid,
            source=train_source,
            upc=instance.upc,
            source_image_id=instance.pk,
            source_from=2,
            score=1.0,
        )
        return Response(util.wrap_ret([]), status=status.HTTP_200_OK)

    # def list(self, request, *args, **kwargs):
    #     queryset = self.filter_queryset(self.get_queryset())
    #
    #     page = self.paginate_queryset(queryset)
    #     if page is not None:
    #         serializer = self.get_serializer(page, many=True)
    #         # logger.info(len(serializer.data))
    #         groundtruthpk_to_cnt = {}
    #         remove_indexes = []
    #         for i in range(len(serializer.data)):
    #             index = len(serializer.data)-i-1
    #             one = serializer.data[index]
    #             groundtruthpk = one['image_ground_truth']['pk']
    #             if groundtruthpk in groundtruthpk_to_cnt:
    #                 if groundtruthpk_to_cnt[groundtruthpk] >= 5:
    #                     remove_indexes.append(index)
    #                 else:
    #                     groundtruthpk_to_cnt[groundtruthpk] += 1
    #             else:
    #                 groundtruthpk_to_cnt[groundtruthpk] = 1
    #         # logger.info(len(remove_indexes))
    #         for index in remove_indexes:
    #             serializer._data.pop(index) # _data is the real data
    #         # logger.info(len(serializer.data))
    #         return self.get_paginated_response(serializer.data)
    #
    #     serializer = self.get_serializer(queryset, many=True)
    #     return Response(serializer.data)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        try:
            os.remove(instance.source.path)
        except:
            pass
        self.perform_destroy(instance)
        return Response(util.wrap_ret(None), status=status.HTTP_200_OK)


class ImageViewSet(DefaultMixin, mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                   viewsets.GenericViewSet):
    queryset = Image.objects.order_by('-id')
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):

        # TODO 只保留deviceid=485进入存储和检测环节
        if request.data['deviceid'] != '485':
            return Response([], status=status.HTTP_201_CREATED)

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

        # hand_detect = HandDetect(serializer.instance.source.path, debug_type=1)
        # is_hand = hand_detect.detect()
        # if is_hand:
        #     serializer.instance.is_hand = True
        #     serializer.instance.save()
        #     logger.info('[{}]detect result: {}'.format(serializer.instance.deviceid, 'has hand'))
        #     return Response([], status=status.HTTP_201_CREATED, headers=headers)
        if device.deviceid not in ['485','200']:
            return Response([],status=status.HTTP_201_CREATED, headers=headers)

        scores, upcs = self.ai_detect(serializer)

        if len(upcs)>0 and upcs[0] == 'hand':
            serializer.instance.is_hand = True
            serializer.instance.save()
        ret = []
        if True: #device.deviceid in ['485','200'] or device.state >= common.DEVICE_STATE_COMMERCIAL:
            # 没有商用的不返回结果
            # upc_to_scores = {}
            # weight = 0.5
            # image_qs = Image.objects.filter(identify=serializer.instance.identify).order_by('-id')
            # for image in image_qs:
            #     image_result_qs = image.image_results.all()
            #     for image_result in image_result_qs:
            #         upc = image_result.upc
            #         score = image_result.score
            #         if upc in upc_to_scores:
            #             upc_to_scores[upc] = upc_to_scores[upc]*(1-weight) + score*weight
            #         else:
            #             upc_to_scores[upc] = score
            #     weight -= 0.05 #前面次数衰减
            #     if weight <= 0:
            #         break
            #
            # upcs, scores = sort_upc_to_scores(upc_to_scores)
            # logger.info(scores)
            for i in range(len(upcs)):
                # TODO 只推荐正确的
                if scores[i]>=0.85 and upcs[i] not in ['hand', 'bag']:
                    ret.append(
                        {
                            'upc': upcs[i],
                            'score': scores[i],
                        }
                    )
        # elif device.deviceid == '36':
        #     ret = [
        #         {'upc': '2000000000103', 'score': 0.99},
        #         {'upc': '2000000000097', 'score': 0.99},
        #         {'upc': '2000000000093', 'score': 0.99}
        #     ]
        # elif device.deviceid == '3061':# 苹果
        #     ret = [
        #         {'upc': '2000000001540', 'score': 0.07},
        #         {'upc': '2000000001598', 'score': 0.25},
        #         {'upc': '3287583', 'score': 0.04},
        #     ]
        # elif device.deviceid == '3062':# 香蕉
        #     ret = [
        #         {'upc': '2000000001541', 'score': 0.95},
        #         {'upc': '3960271', 'score': 0.03},
        #         {'upc': '3283458', 'score': 0.01},
        #     ]

        logger.info('[{}]detect result: {}'.format(serializer.instance.deviceid, ret))
        return Response(ret, status=status.HTTP_201_CREATED, headers=headers)

    def ai_detect(self, serializer):
        upcs = []
        scores = []
        # 检测阶段
        if serializer.instance.deviceid in common.good_neighbour_bind_deviceid_list:
            # 好邻居联合计算
            last_normal_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).filter(
                deviceid__in=common.good_neighbour_bind_deviceid_list).exclude(action='TC').order_by('-id')
            if len(last_normal_train_qs) > 0:
                logger.info('[{}]begin detect image:{}'.format(serializer.instance.deviceid, serializer.instance.identify))
                last_train = last_normal_train_qs[0]
                last_normal_train_model = \
                TrainModel.objects.filter(train_action_id=last_train.pk).exclude(model_path='').order_by('-id')[0]
                detector = imagedetection.ImageDetectorFactory.get_static_detector(
                    last_normal_train_model)
                upcs, scores = detector.detect(serializer.instance)
                ImageTrainModel.objects.create(
                    train_model_id=last_normal_train_model.pk,
                    image_id=serializer.instance.pk
                )
        else:
            last_normal_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).filter(
                deviceid=serializer.instance.deviceid).exclude(action='TC').order_by('-id')
            if len(last_normal_train_qs) > 0:
                logger.info('[{}]begin detect image:{}'.format(serializer.instance.deviceid, serializer.instance.identify))
                last_train = last_normal_train_qs[0]
                last_normal_train_model = \
                TrainModel.objects.filter(train_action_id=last_train.pk).exclude(model_path='').order_by('-id')[0]
                detector = imagedetection.ImageDetectorFactory.get_static_detector(
                    last_normal_train_model)
                upcs, scores = detector.detect(serializer.instance)
                ImageTrainModel.objects.create(
                    train_model_id=last_normal_train_model.pk,
                    image_id=serializer.instance.pk
                )

            # last_tc_train_qs = TrainAction.objects.filter(state=common.TRAIN_STATE_COMPLETE).filter(
            #     deviceid=serializer.instance.deviceid).filter(action='TC').filter(
            #     complete_time__gt=last_normal_train_model.create_time).order_by('-id')
            # if len(last_tc_train_qs) > 0:
            #     last_tc_train = last_tc_train_qs[0]
            #     last_tc_train_model = \
            #     TrainModel.objects.filter(train_action_id=last_tc_train.pk).exclude(model_path='').order_by('-id')[0]
            #     detector2 = imagedetection.ImageDetectorFactory.get_static_detector(
            #         last_tc_train_model)
            #     upcs2, scores2 = detector2.detect(serializer.instance)
            #     ImageTrainModel.objects.create(
            #         train_model_id=last_tc_train_model.pk,
            #         image_id=serializer.instance.pk
            #     )
            #     # 联合计算
            #     upc_to_scores = {}
            #     for i in range(len(upcs)):
            #         if upcs[i] in upc_to_scores:
            #             upc_to_scores[upcs[i]] = upc_to_scores[upcs[i]] * 0.5 + scores[i] * 0.5
            #         else:
            #             upc_to_scores[upcs[i]] = scores[i]
            #     for i in range(len(upcs2)):
            #         if upcs2[i] in upc_to_scores:
            #             upc_to_scores[upcs2[i]] = upc_to_scores[upcs2[i]] * 0.5 + scores2[i] * 0.5
            #         else:
            #             upc_to_scores[upcs2[i]] = scores2[i]
            #
            #     upcs, scores = sort_upc_to_scores(upc_to_scores)

        # 输出结果
        for i in range(len(upcs)):
            if i < 5:  # 不超过5个
                ImageResult.objects.create(
                    image_id=serializer.instance.pk,
                    upc=upcs[i],
                    score=scores[i]
                )
        return scores, upcs


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
                    logger.info('[{}]update image ground truth:{},{}'.format(serializer.data['deviceid'], serializer.data['identify'], serializer.data['upc']))
                except Exception as e:
                    logger.error(e)
                    raise e
            else:
                logger.error(e)
                raise e
        else:
            logger.info('[{}]create image ground truth:{},{}'.format(serializer.data['deviceid'], serializer.data['identify'], serializer.data['upc']))

        images = Image.objects.filter(identify=serializer.instance.identify).filter(deviceid=serializer.instance.deviceid)
        truth_image_result_cnt = 0
        false_image_result_cnt = 0
        for image in images:
            truth_image_result_qs = image.image_results.filter(upc=serializer.instance.upc).filter(score__gt=0.85)
            if len(truth_image_result_qs)>0:
                truth_image_result_cnt += 1
            false_image_result_qs = image.image_results.exclude(upc=serializer.instance.upc).exclude(upc='hand').exclude(upc='bag').filter(score__gt=0.85)
            if len(false_image_result_qs)>0:
                false_image_result_cnt += 1
            image.image_ground_truth=serializer.instance
            image.upc = serializer.instance.upc
            image.save()

        serializer.instance.cnt = len(images)
        if truth_image_result_cnt+false_image_result_cnt > 0:
            if truth_image_result_cnt>0:
                serializer.instance.truth_rate = 1.0
                serializer.instance.precision = 1.0
            else:
                serializer.instance.truth_rate = 0.0
                serializer.instance.precision = 0.0
        else:
            serializer.instance.truth_rate = 1.0
            serializer.instance.precision = 0.0
        serializer.instance.save()

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

class UpcBindViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = UpcBind.objects.order_by('-id')
    serializer_class = UpcBindSerializer
    filter_backends = (DjangoFilterBackend,)
    filter_fields = ('deviceid',)


class TrainImageViewSet(DefaultMixin, viewsets.ModelViewSet):
    # TODO 为了临时显示，去掉手和袋的显示
    queryset = TrainImage.objects.filter(special_type=0).order_by('-id')
    serializer_class = TrainImageSerializer
    filter_backends = (DjangoFilterBackend,)
    filter_fields = ('deviceid', 'upc')

    @action(methods=['get'], detail=False)
    def device_list(self, request):
        devices = TrainImage.objects.values('deviceid').distinct()
        ret = []
        for deviceid in devices:
            ret.append(deviceid['deviceid'])
        return Response(ret)

    @action(methods=['get'], detail=False)
    def device_to_precision(self, request):
        devices = TrainImage.objects.values('deviceid').distinct()
        ret = {}
        for deviceid in devices:
            try:
                device = Deviceid.objects.get(deviceid=deviceid['deviceid'])
            except Deviceid.DoesNotExist as e:
                device = Deviceid.objects.create(
                    deviceid=deviceid['deviceid'],
                )
            precision_qs = device.device_precisions.order_by('-id')
            if len(precision_qs)>0:
              ret[deviceid['deviceid']] = precision_qs[0].truth_rate
            else:
              ret[deviceid['deviceid']] = 0.0
        return Response(ret)

    @action(methods=['get'], detail=False)
    def upc_list(self, request):
        if 'deviceid' in request.query_params:
            deviceid = request.query_params['deviceid']
            upcs = TrainImage.objects.filter(deviceid=deviceid).values('upc').distinct()
        else:
            upcs = TrainImage.objects.values('upc').distinct()
        ret = []
        for upc in upcs:
            ret.append(upc['upc'])
        return Response(ret)

    @action(methods=['put'], detail=True)
    def set_special_type(self, request, pk=None):
        instance = self.get_object()
        if 'type' not in request.query_params:
            type = 0
        else:
            type = int(request.query_params['type'])

        if type>0:
            if type==1:
                special_type_name = 'hand'
            elif type==2:
                special_type_name = 'bag'
            special_source_dir = '{}/{}'.format(common.get_dataset_dir(True),special_type_name)
            special_source_path = '{}/{}_{}'.format(special_source_dir, special_type_name, os.path.basename(instance.source.path))
            shutil.copy(instance.source.path, special_source_path)
        else:
            if instance.special_type == 1:
                special_type_name = 'hand'
            elif instance.special_type == 2:
                special_type_name = 'bag'
            special_source_dir = '{}/{}'.format(common.get_dataset_dir(True), special_type_name)
            special_source_path = '{}/{}_{}'.format(special_source_dir, special_type_name,
                                                    os.path.basename(instance.source.path))
            if os.path.isfile(special_source_path):
                os.remove(special_source_path)

        instance.special_type = type
        instance.save()
        return Response(util.wrap_ret([]), status=status.HTTP_200_OK)


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
        try:
            os.remove(instance.source.path)
        except:
            pass
        self.perform_destroy(instance)
        return Response(util.wrap_ret(None), status=status.HTTP_200_OK)


class TrainActionViewSet(DefaultMixin, viewsets.ModelViewSet):
    queryset = TrainAction.objects.order_by('-id')
    serializer_class = TrainActionSerializer
    filter_backends = (DjangoFilterBackend,)
    filter_fields = ('deviceid',)

class TrainModelViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = TrainModel.objects.order_by('-id')
    serializer_class = TrainModelSerializer

class TaskLogViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = TaskLog.objects.order_by('-id')
    serializer_class = TaskLogSerializer

class CreateTrain(APIView):
    def get(self, request):
        if 'action' not in request.query_params:
            action = 'TA'
        else:
            action = request.query_params['action']
        deviceid = request.query_params['deviceid']
        if deviceid == 'bag':
            from goods2.cron import do_create_train_bag

            train_action = do_create_train_bag(action,deviceid)
            logger.info('[{}]create_train_bag by menu: {}'.format(deviceid, action))
            return Response(util.wrap_ret(None), status=status.HTTP_201_CREATED)

        if deviceid in common.good_neighbour_bind_deviceid_list:
            # 好邻居联合计算
            waiting_ta_tf = TrainAction.objects.exclude(action='TC').filter(deviceid__in=common.good_neighbour_bind_deviceid_list).filter(
                state__lte=common.TRAIN_STATE_WAITING)
            if len(waiting_ta_tf) == 0:
                from goods2.cron import do_create_train_bind

                train_action = do_create_train_bind(action, deviceid, None, common.good_neighbour_bind_deviceid_list)
                logger.info('[{}]create_train_bind by menu: {}'.format(deviceid, action))
                return Response(util.wrap_ret(None), status=status.HTTP_201_CREATED)
            else:
                return Response(status=status.HTTP_204_NO_CONTENT)
        else:
            waiting_ta_tf = TrainAction.objects.exclude(action='TC').filter(deviceid=deviceid).filter(state__lte=common.TRAIN_STATE_WAITING)
            if len(waiting_ta_tf) == 0:
                from goods2.cron import do_create_train

                train_action = do_create_train(action, deviceid, None)
                logger.info('[{}]create_train by menu: {}'.format(deviceid, action))
                return Response(util.wrap_ret(None), status=status.HTTP_201_CREATED)
            else:
                return Response(status=status.HTTP_204_NO_CONTENT)


class TestTrain(APIView):
    def get(self, request):
        action = 'TA'
        doing_ta_tf = TrainAction.objects.exclude(action='TC').filter(
            state__lte=common.TRAIN_STATE_TRAINING)
        if len(doing_ta_tf) == 0:
            from goods2.cron import do_test_train

            train_action = do_test_train(action)
            logger.info('begin test_train')
            return Response(util.wrap_ret(None), status=status.HTTP_201_CREATED)
        else:
            return Response(status=status.HTTP_204_NO_CONTENT)


class ClearData(APIView):
    def get(self, request):
        # deviceid = request.query_params['deviceid']
        # train_image_qs = TrainImage.objects.filter(source_image_id__gt=0).filter(deviceid=deviceid).order_by('id')
        # for train_image in train_image_qs:
        #     os.remove(train_image.source.path)
        #     train_image.delete()

        image_qs = Image.objects.filter(image_ground_truth=None)
        for image in image_qs:
          if os.path.isfile(image.source.path):
            logger.info('delete image: {}'.format(image.source.path))
            os.remove(image.source.path)
          for result in image.image_results.all():
            result.delete()
          image.delete()

        return Response([], status=status.HTTP_201_CREATED)
