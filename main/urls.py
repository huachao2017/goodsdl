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
from django.conf.urls import url
from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings
from goods.urls import urlpatterns as goods_urlpatterns
from goods2.urls import urlpatterns as goods2_urlpatterns
from goodscf.urls import urlpatterns as goodscf_urlpatterns
from face.urls import urlpatterns as face_urlpatterns
from arm.urls import urlpatterns as arm_urlpatterns
from track.urls import urlpatterns as track_urlpatterns


urlpatterns = [
    url(r'^admin/', admin.site.urls),
]+goods_urlpatterns+goods2_urlpatterns+goodscf_urlpatterns+face_urlpatterns+arm_urlpatterns+track_urlpatterns+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
