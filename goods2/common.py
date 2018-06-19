from django.conf import settings
import os

DATASET_DIR = settings.DATASET_DIR_NAME + '/goods2'
DETECT_DIR = settings.DETECT_DIR_NAME + '/goods2'
TRAIN_DIR = os.path.join(settings.TRAIN_ROOT, 'goods2')
MODEL_DIR = os.path.join(settings.MODEL_ROOT, 'goods2')