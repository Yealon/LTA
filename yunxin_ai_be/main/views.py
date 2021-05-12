from django.http import HttpResponse, Http404
from django.shortcuts import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_jwt.utils import jwt_decode_handler
from .models import *
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.backends import ModelBackend
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth import get_user_model, authenticate
from django.db.models import Q
from rest_framework_jwt.settings import api_settings
from .utils import TokenAuth
import bytedurllib


def index(request):
    """
    An easy example

    :param request:
    :return:
    """
    return HttpResponse("{'name':'刘屹林', 'student_id':'2001210637'}")


def user_info(request):
    """
    An easy example

    :param request:
    :return:
    """
    return HttpResponse('''{
        permissions: ["auth", "auth/testPage", "auth/authPage", "auth/authPage/edit", "auth/authPage/visit"],
        role: "系统管理员",
        roleType: 1,
        uid: 1,
        userName: "系统管理员",
      }''')


def get_menu(request):
    return HttpResponse(
        '[{"key":"/app/smenu","title":"异步菜单","icon":"api","subs":[{"key":"/app/smenu/sub1","title":"异步子菜单1","component":"Sub1"},{"key":"/app/smenu/sub2","title":"异步子菜单2","component":"Sub2"}]}]'
    )


def operate_session(request):
    # 设置session
    request.session['name'] = 'laowang'
    request.session.get('name', 'laoli')
    return HttpResponse("OK")


class OrderAPIView(APIView):  # 登录才能
    authentication_classes = [TokenAuth, ]

    # 权限控制
    # 自己写权限控制，不用这个智障框架了
    # permission_classes = [IsAuthenticated, ]

    def get(self, request, *args, **kwargs):
        # tos = bytedtos.Client("<bucket>", "<access_key>")
        # value = b'hello world'
        # tos.put_object("test-key", value)
        # print(tos.get_object("test-key").data, value)
        # print(tos.head_object("test-key").size, 11)
        # print(tos.get_object_range("test-key", 1, 3).data, b"ell")
        # print(tos.delete_object("test-key"), True)
        #
        # try:
        #     tos.get_object("test-key")
        # except bytestos.TosException as e:
        #     print(e.code, 404)
        #
        # payload_id = tos.init_upload("test-key").upload_id
        # n = 3
        # largeValue = value * 1000 * 1000
        # for i in range(n):
        #     tos.upload_part("test-key", payload_id, "%d" % i, largeValue)
        # tos.complete_upload("test-key", payload_id, list(map(lambda u: "%d" % u, range(n))))
        # print(tos.get_object("test-key").data == largeValue * n)
        #
        # tos.put_object("test-key-file", open("../tests/data", "r"))
        # print(tos.get_object("test-key-file").data, b'hello world')
        #
        # tos.stream = True
        # resp = tos.get_object("test-key-file").raw
        # print(resp.read(3), b'hel')
        # print(resp.read(2), b'lo')
        return Response('测试Tos')


class UserInfoAPIView(APIView):
    authentication_classes = [TokenAuth, ]

    # 权限控制
    # permission_classes = [IsAuthenticated,]
    def get(self, request, *args, **kwargs):
        data = {
            "building_num": 1,
            "other_stu": [2]
        }
        building_num = data['building_num']
        request_user_auth = jwt_decode_handler(request.META.get("HTTP_TOKEN"))

        other_stu = []
        if data['other_stu']:
            other_stu = data['other_stu']
        stus = other_stu[:]
        stus.append(request_user_auth['user_id'])

        # 都没寝室
        users = TbUser.objects.filter(dorm_id=None).filter(id__in=stus)
        if users.__len__() < len(stus):
            return JsonResponse({"code": 501, "message": "student status error"})
        for user in users:
            if user.sex != users[0].sex:
                return JsonResponse({"code": 502, "message": "student sex error"})

        dorms = TbDorm.objects.filter(num_empty__gte=users.__len__()).filter(state=1).filter(id=building_num)
        if dorms.__len__() == 0:
            return JsonResponse({"code": 501, "message": "no dorm left"})
        # 选中dorms[0]
        final_users = TbUser.objects.filter(id__in=stus).update(dorm_id=dorms[0].id)
        final_dorm = TbDorm.objects.filter(id=dorms[0].id).update(num_empty=dorms[0].num_empty - users.__len__())

        return JsonResponse({"code": 200, "message": "success", "data": {"dorm": dorms[0].name}})

    def post(self, request, *args, **kwargs):
        data = request.data
        building_num = data['building_num']
        request_user_auth = jwt_decode_handler(request.META.get("HTTP_TOKEN"))

        other_stu = []
        if data['other_stu']:
            other_stu = data['other_stu']
        stus = other_stu[:]
        stus.append(request_user_auth['user_id'])

        # 都没寝室
        users = TbUser.objects.filter(dorm_id=None).filter(id__in=stus)
        if users.__len__() < len(stus):
            return JsonResponse({"code": 501, "message": "student status error"})
        for user in users:
            if user.sex != users[0].sex:
                return JsonResponse({"code": 502, "message": "student sex error"})

        dorms = TbDorm.objects.filter(num_empty__gte=users.__len__()).filter(state=1).filter(id=building_num)
        if dorms.__len__() == 0:
            return JsonResponse({"code": 501, "message": "no dorm left"})
        # 选中dorms[0]
        final_users = TbUser.objects.filter(id__in=stus).update(dorm_id=dorms[0].id)
        final_dorm = TbDorm.objects.filter(id=dorms[0].id).update(num_empty=dorms[0].num_empty - users.__len__())

        return JsonResponse({"code": 200, "message": "success", "data": {"dorm": dorms[0].name}})


class LegalAnalyseAPIView(APIView):
    authentication_classes = [TokenAuth, ]

    # 权限控制
    # permission_classes = [IsAuthenticated,]
    # 用GET可能导致参数最大长度限制
    def post(self, request, *args, **kwargs):
        data = request.data
        legal_text = data['legal_text']
        request_user_auth = jwt_decode_handler(request.META.get("HTTP_TOKEN"))

        if request_user_auth['can_use']:
            pass
        else:  # 超期了，或者本来就没权限，不能使用这个接口了
            return JsonResponse({"code": 510, "message": "no user auth for this interface", "data": {}})

        params = {'legal_text': legal_text}

        # req_url = "http://10.227.91.52/legal/text" # 内网连接开发机
        # req_url = "http://10.227.91.56/legal/text"
        # req_url = "http://81.70.98.70/legal/text" # 公网服务器
        req_url = "http://81.70.98.71/legal/text"

        res_text = bytedurllib.Request(url=req_url, data=bytedurllib.urlencode(params))
        return JsonResponse({"code": 200, "message": "success", "data": {"legal_tag": res_text.tags}})


class Index(APIView):
    authentication_classes = [TokenAuth, ]

    def index1(request):
        return JsonResponse({"index": "ok"})


class CustomBackend(ModelBackend):
    def authenticate(self, request, phone_num=None, password=None, **kwargs):
        try:
            user = TbUser.objects.filter(username=kwargs['username']).filter(password=password)[0]
            return user
        except Exception as e:
            return None
