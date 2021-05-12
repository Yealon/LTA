# author: Yilin6
# datetime: 2020/11/13 10:01

from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, AbstractUser
from .UserManage import UserManager

""" User Related Tables

========================================

"""


class TbDorm(models.Model):
    """ The dorm for students
    """
    id = models.AutoField(primary_key=True)
    build_num = models.IntegerField(unique=True)
    name = models.CharField(max_length=20)
    num_beds = models.IntegerField(default=4)
    num_empty = models.IntegerField(default=0)
    name = models.CharField(max_length=20)
    state = models.IntegerField(default=1)

    def __str__(self):
        return self.name


class TbUser(AbstractBaseUser, PermissionsMixin):
    """ Common user.
    """

    id = models.AutoField(primary_key=True)
    long_id = models.CharField(max_length=64)
    username = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=64)
    sex = models.IntegerField(default=1)
    points = models.IntegerField(default=0)
    head_pic = models.CharField(max_length=100)
    from_type = models.IntegerField(default=0)
    phone_num = models.CharField(max_length=20)
    email = models.CharField(max_length=50)
    wechat = models.CharField(max_length=50)
    dorm = models.ForeignKey(TbDorm, on_delete=models.CASCADE, default=1)

    # students related
    class_num = models.IntegerField(default=0)
    grade_num = models.IntegerField(default=0)
    # state related - 0: usable, 1: disable
    state = models.IntegerField(default=1)
    phone_num_state = models.IntegerField(default=0)
    email_state = models.IntegerField(default=0)
    wechat_state = models.IntegerField(default=0)
    # time related
    create_time = models.DateTimeField('date created')
    birth_time = models.DateTimeField('birthday')
    last_login = models.DateTimeField('last login time')
    # delete related
    is_deleted = models.IntegerField(default=0)
    delete_time = models.DateTimeField('delete time')

    # jwt
    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)
    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['id', 'email']

    objects = UserManager()

    def __str__(self):
        return self.username

    def has_perm(self, perm, obj=None):
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        # Simplest possible answer: Yes, always
        return True


class TrUserDorm(models.Model):
    """ The relation between users and dorms.
    R<U, D>: U is living in D.
    """
    id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(TbUser, on_delete=models.CASCADE)
    dorm_id = models.ForeignKey(TbDorm, on_delete=models.CASCADE)
    state = models.IntegerField(default=1)


class TbRole(models.Model):
    """ The roles the common user play.
    """
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=20, unique=True)
    state = models.IntegerField(default=1)
    comment = models.CharField(max_length=50)
    users = models.ManyToManyField(TbUser)

    def __str__(self):
        return self.name


class TrUserRole(models.Model):
    """ The relation between users and roles.
    R<U, R>: U can play as R.
    """
    id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(TbUser, on_delete=models.CASCADE)
    role_id = models.ForeignKey(TbRole, on_delete=models.CASCADE)
    state = models.IntegerField(default=1)


class TbLegalCauseData(models.Model):
    """ The data of forecast of legal cause
    """
    id = models.AutoField(primary_key=True)
    cause = models.CharField(max_length=3000)
    result = models.CharField(max_length=50)
    state = models.IntegerField(default=1)

    def __str__(self):
        return self.cause
