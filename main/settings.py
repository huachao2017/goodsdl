"""
Django settings for dlserver project.

Generated by 'django-admin startproject' using Django 1.11.6.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

import os
import socket

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '*nqbimelq553&@0q7&ydkwe*2fbmb6#1%*xibu@-@%u0u@+dw@'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

if DEBUG:
    ALLOWED_HOSTS = ['*']
else:
    ALLOWED_HOSTS = ['train.fastxbox.cn', 'ai.fastxbox.cn']

# Application definition

INSTALLED_APPS = [
    'django_crontab',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'django_filters',
    'goods',
    'goods2',
    'goodscf',
    'face',
    'arm',
    'track',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'main.urls'

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'main.pagination.MyPageNumberPagination',
    'PAGE_SIZE': 50,
    'DEFAULT_FILTER_BACKENDS': ('django_filters.rest_framework.DjangoFilterBackend',)
}

CRONJOBS = [
    ('0 23 * * *', 'goods2.cron.check_device', '>> {} 2>&1'.format(os.path.join(BASE_DIR, "logs", 'cron.log'))),
    # ('*/1 * * * *', 'goods2.cron.transfer_sample', '>> {} 2>&1'.format(os.path.join(BASE_DIR, "logs", 'cron.log'))),
    ('*/10 * * * *', 'goods2.cron.create_train', '>> {} 2>&1'.format(os.path.join(BASE_DIR, "logs", 'cron.log'))),
    ('*/10 * * * *', 'goods2.cron.execute_train', '>> {} 2>&1'.format(os.path.join(BASE_DIR, "logs", 'cron.log'))),
    ('*/10 * * * *', 'goods2.cron.check_train', '>> {} 2>&1'.format(os.path.join(BASE_DIR, "logs", 'cron.log'))),
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'main.wsgi.application'

# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

# def get_host_ip():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.connect(('8.8.8.8', 80))
#         ip = s.getsockname()[0]
#     finally:
#         s.close()
#
#     return ip
#
# my_host_ip =  get_host_ip()
#
# # TODO need remove
# if my_host_ip == '192.168.1.73':
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'goodsdl',
        'USER': 'root',
        'PASSWORD': 'iShehui',
        'HOST': '192.168.1.62',
        'PORT': '3306',
        'CHARSET': 'utf8',
        'COLLATION': 'utf8_general_ci',
    }
}
# else:
#     DATABASES = {
#         'default': {
#             'ENGINE': 'django.db.backends.sqlite3',
#             'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
#         }
#     }

# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True

# USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

STATIC_URL = '/static/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

DATASET_DIR_NAME = 'dataset'
DETECT_DIR_NAME = 'images'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },  # 针对 DEBUG = True 的情况
    },
    'formatters': {
        'standard': {
            # 'format': '%(levelname)s %(asctime)s %(pathname)s %(lineno)d: %(message)s'
            'format': '%(levelname)s %(asctime)s: %(message)s'
        },  # 对日志信息进行格式化，每个字段对应了日志格式中的一个字段，更多字段参考官网文档，我认为这些字段比较合适，输出类似于下面的内容
        # INFO 2016-09-03 16:25:20,067 /home/ubuntu/mysite/views.py views.py views get 29: some info...
    },
    'handlers': {
        'request_file_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'debug.log'),
            'maxBytes': 1024 * 1024 * 50,  # 文件大小
            'backupCount': 50,  # 备份份数
            'formatter': 'standard'
        },  # 用于文件输出
        'detect_file_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'detect.log'),
            'maxBytes': 1024 * 1024 * 50,  # 文件大小
            'backupCount': 50,  # 备份份数
            'formatter': 'standard'
        },  # 用于文件输出
        'request2_file_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'debug2.log'),
            'maxBytes': 1024 * 1024 * 50,  # 文件大小
            'backupCount': 50,  # 备份份数
            'formatter': 'standard'
        },  # 用于文件输出
        'detect2_file_handler': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'detect2.log'),
            'maxBytes': 1024 * 1024 * 50,  # 文件大小
            'backupCount': 50,  # 备份份数
            'formatter': 'standard'
        },  # 用于文件输出
        'face_file_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'face.log'),
            'maxBytes': 1024 * 1024 * 50,  # 文件大小
            'backupCount': 50,  # 备份份数
            'formatter': 'standard'
        },  # 用于文件输出
        'classify_file_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'classify.log'),
            'maxBytes': 1024 * 1024 * 50,  # 文件大小
            'backupCount': 50,  # 备份份数
            'formatter': 'standard'
        },  # 用于文件输出
        'dataset_file_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'dataset.log'),
            'maxBytes': 1024 * 1024 * 50,  # 文件大小
            'backupCount': 50,  # 备份份数
            'formatter': 'standard'
        },  # 用于文件输出
        'console': {
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'cron_log_file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, "logs", 'cron.log'),
            'formatter': 'standard'
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'request_file_handler'],
            'level': 'DEBUG',
            'propagate': True  # 是否继承父类的log信息
        },  # handlers 来自于上面的 handlers 定义的内容
        'detect': {
            'handlers': ['console', 'detect_file_handler'],
            'level': 'DEBUG',
            'propagate': True  # 是否继承父类的log信息
        },  # handlers 来自于上面的 handlers 定义的内容
        'django2': {
            'handlers': ['console', 'request2_file_handler'],
            'level': 'DEBUG',
            'propagate': True  # 是否继承父类的log信息
        },  # handlers 来自于上面的 handlers 定义的内容
        'detect2': {
            'handlers': ['console', 'detect2_file_handler'],
            'level': 'DEBUG',
            'propagate': True  # 是否继承父类的log信息
        },  # handlers 来自于上面的 handlers 定义的内容
        'face': {
            'handlers': ['console', 'face_file_handler'],
            'level': 'DEBUG',
            'propagate': True  # 是否继承父类的log信息
        },  # handlers 来自于上面的 handlers 定义的内容
        'classify': {
            'handlers': ['console', 'classify_file_handler'],
            'level': 'DEBUG',
            'propagate': True  # 是否继承父类的log信息
        },  # handlers 来自于上面的 handlers 定义的内容
        'dataset': {
            'handlers': ['console', 'dataset_file_handler'],
            'level': 'DEBUG',
            'propagate': True  # 是否继承父类的log信息
        },  # handlers 来自于上面的 handlers 定义的内容
        'cron': {  # The namespace of the logger above
            'handlers': ['cron_log_file'],
            'level': 'DEBUG',
            'propagate': True,
        },

    }
}

# train setting
TRAIN_ROOT = os.path.join(BASE_DIR, 'train')
MODEL_ROOT = os.path.join(BASE_DIR, 'dl', 'model')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"