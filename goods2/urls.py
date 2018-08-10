"""dlserver URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter

from goods2 import views

router = DefaultRouter()
router.register(r'device', views.DeviceidViewSet)
router.register(r'devicetrain', views.DeviceidTrainViewSet)
router.register(r'image', views.ImageViewSet)
router.register(r'imagegroundtruth', views.ImageGroundTruthViewSet)
router.register(r'trainimage', views.TrainImageViewSet)

router.register(r'trainaction', views.TrainActionViewSet)
router.register(r'trainmodel', views.TrainModelViewSet)
router.register(r'tasklog', views.TaskLogViewSet)
urlpatterns = [
    url(r'^api2/', include(router.urls))
]
