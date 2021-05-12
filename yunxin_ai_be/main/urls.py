"""testpy1 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.urls import path
from . import views
from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token

urlpatterns = [
    path('user_info', views.user_info, name='user_info'),
    path('get_menu', views.get_menu, name='get_menu'),
    path(r"login", obtain_jwt_token),
    path(r'index1', views.Index.index1, name='index'),
    path(r'get1', views.OrderAPIView.as_view(), name='index1'),
    path(r'user', views.UserInfoAPIView.as_view(), name='user'),
    path(r'legal_text', views.LegalAnalyseAPIView.as_view(), name='legal'),
    url(r'^operate_session/$', views.operate_session),
]
