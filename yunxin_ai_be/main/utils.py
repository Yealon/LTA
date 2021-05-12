from rest_framework.exceptions import AuthenticationFailed
from rest_framework_jwt.authentication import BaseJSONWebTokenAuthentication
from rest_framework_jwt.serializers import VerifyJSONWebTokenSerializer
from .models import *

request_url_list = [
    '/main/get0',
    '/main/user',
    '/main/get2',
    '/main/get3',
    '/main/get4',
]

"""
自定义jwt认证成功返回数据
:token  返回的jwt
:user   当前登录的用户信息[对象]
:request 当前本次客户端提交过来的数据
:role 角色
"""


def jwt_response_payload_handler(token, user=None, request=None, role=None):
    return {
        "authenticated": 'true',
        'id': user.id,
        "role": role,
        'username': user.get_username(),
        'email': user.email,
        'token': token,
    }


class TokenAuth(BaseJSONWebTokenAuthentication):
    def authenticate(self, request):
        token = {"token": None}
        # print(request.META.get("HTTP_TOKEN"))
        token["token"] = request.META.get('HTTP_TOKEN')
        valid_data = VerifyJSONWebTokenSerializer().validate(token)
        user = valid_data['user']
        if user:
            roles = TbUser.objects.get(username=user.username).tbrole_set.all()
            if roles.__len__() > 0:
                for auth_nodes in roles[0].comment.split("."):
                    trav_url = request_url_list[int(auth_nodes)]
                    if trav_url and trav_url.count(
                            request.get_full_path()) > 0 and \
                            trav_url.index(request.get_full_path()) == 0:
                        return
            raise AuthenticationFailed({'msg': '认证成功，授权失败', 'code': '211'})
        else:
            raise AuthenticationFailed({'msg': '认证失败', 'code': '011'})
