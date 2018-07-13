from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import mixins
from rest_framework import viewsets

from goodscf.serializers import *
from goodscf.models import UserGoods, Users

# Create your views here.

class DefaultMixin:
    paginate_by = 25
    paginate_by_param = 'page_size'
    max_paginate_by = 100

class UsersViewSet(DefaultMixin, viewsets.ReadOnlyModelViewSet):
    queryset = Users.objects.order_by('-id')
    serializer_class = UsersSerializer

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        ret = {'recommend':[], 'source':[]}

        user_goods_est_qs = UserGoods.objects.filter(openid=instance.openid).filter(r_ui=0).order_by('-est')[:5]
        for i in range(len(user_goods_est_qs)):
            ret['recommend'].append({
                'goods_name':user_goods_est_qs[i].goods.goods_name,
                'rating':user_goods_est_qs[i].est,
            })

        user_goods_source_qs = UserGoods.objects.filter(openid=instance.openid).filter(r_ui__gt=0).order_by('-r_ui')
        for i in range(len(user_goods_source_qs)):
            ret['source'].append({
                'goods_name':user_goods_source_qs[i].goods.goods_name,
                'rating':user_goods_source_qs[i].r_ui,
                'est': user_goods_source_qs[i].est,
            })
        return Response(ret)

class PredictUser(APIView):
    def get(self, request):
        openid = request.query_params['openid']
        ret = {'recommend':[], 'source':[]}

        user_goods_est_qs = UserGoods.objects.filter(openid=openid).filter(r_ui=0).order_by('-est')[:5]
        for i in range(len(user_goods_est_qs)):
            ret['recommend'].append({
                'goods_name':user_goods_est_qs[i].goods.goods_name,
                'rating':user_goods_est_qs[i].est,
            })

        user_goods_source_qs = UserGoods.objects.filter(openid=openid).filter(r_ui__gt=0).order_by('-r_ui')
        for i in range(len(user_goods_source_qs)):
            ret['source'].append({
                'goods_name':user_goods_source_qs[i].goods.goods_name,
                'rating':user_goods_source_qs[i].r_ui,
                'est': user_goods_source_qs[i].est,
            })


        return Response(ret, status=status.HTTP_200_OK)