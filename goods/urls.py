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

from . import views
from . import views_train

router = DefaultRouter()
# router.register(r'image', views.ImageOldViewSet)
router.register(r'imagenew', views.ImageViewSet)
router.register(r'imagetest', views.ImageTestViewSet)
router.register(r'imagereport', views.ImageReportViewSet)
router.register(r'imageclass', views.ImageClassViewSet)
router.register(r'problemgoods', views.ProblemGoodsViewSet)

router.register(r'trainimage', views_train.TrainImageViewSet)
router.register(r'trainimageonly', views_train.TrainImageOnlyViewSet)
router.register(r'trainimageclass', views_train.TrainImageClassViewSet)
router.register(r'sampleimageclass', views_train.SampleImageClassViewSet)
router.register(r'trainaction', views_train.TrainActionViewSet)
router.register(r'exportaction', views_train.ExportActionViewSet)
router.register(r'stoptrainaction', views_train.StopTrainActionViewSet)
router.register(r'rfidimagecompareaction', views_train.RfidImageCompareActionViewSet)
router.register(r'transactionmetrix', views_train.TransactionMetrixViewSet)
router.register(r'traintask', views_train.TrainTaskViewSet)
router.register(r'clusterstructure', views_train.ClusterStructureViewSet)
router.register(r'clusterevaldata', views_train.ClusterEvalDataViewSet)
router.register(r'clustersamplescore', views_train.ClusterSampleScoreViewSet)
router.register(r'clusterupcscore', views_train.ClusterUpcScoreViewSet)
urlpatterns = [
    url(r'^test', views.Test.as_view()),
    url(r'^api/getsamplecount', views_train.GetSampleCount.as_view()),
    url(r'^api/removeallsample', views_train.RemoveAllSample.as_view()),
    url(r'^api/verifycnt', views.VerifyCnt.as_view()),
    url(r'^api/', include(router.urls))
]
