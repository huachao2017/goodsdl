from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from goodscf.models import UserGoods
# Create your views here.

class Predict(APIView):
    def get(self, request):
        openid = request.query_params['openid']
        ret = {'est':[], 'source':[]}

        user_goods_est_qs = UserGoods.objects.filter(openid=openid).filter(r_ui=0).order_by('-est')[:5]
        for i in range(len(user_goods_est_qs)):
            ret['est'].append({
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