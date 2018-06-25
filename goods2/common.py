from django.conf import settings
import os

PREFIX = 'goods2'
def get_dataset_dir(full_path = False):
    if full_path:
        return os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, PREFIX)
    else:
        return os.path.join(settings.DATASET_DIR_NAME, PREFIX)

def get_detect_dir(full_path = False):
    if full_path:
        return os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, PREFIX)
    else:
        return os.path.join(settings.DETECT_DIR_NAME, PREFIX)

def get_train_path():
    return os.path.join(settings.TRAIN_ROOT, PREFIX)

def get_model_path():
    os.path.join(settings.MODEL_ROOT, PREFIX)