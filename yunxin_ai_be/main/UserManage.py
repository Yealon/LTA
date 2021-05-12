# author: Yilin6
# datetime: 2020/11/20 09:09

from django.contrib.auth.models import AbstractUser, BaseUserManager
import datetime


class UserManager(BaseUserManager):
    use_in_migrations = True

    def create_user(self, username, email=None, password=None, phone_num=None, userID=None, **extra_fields):
        if not username:
            raise ValueError('The given username must be set')
        if not password:
            raise ValueError('The given password must be set')
        if not phone_num:
            raise ValueError('The given phone_num must be set')
        if not email:
            raise ValueError('The given email must be set')
        email = self.normalize_email(email)
        username = self.model.normalize_username(username)
        user = self.model(username=username, email=email, phone_num=phone_num, userID=userID,
                          reg_date=datetime.datetime.now(), **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, phone_num, userID, username, email, password, **extra_fields):
        user = self.create_user(email=email,
                                password=password,
                                username=username,
                                phone_num=phone_num,
                                userID=userID
                                )
        user.is_admin = True
        user.save(using=self._db)
        return user

    def get_by_natural_key(self, username):
        return self.get(username__iexact=username)
