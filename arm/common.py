from django.conf import settings
import os

_STORAGE_PREFIX = 'arm'
def get_dataset_dir(full_path = False):
    if full_path:
        return os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR_NAME, _STORAGE_PREFIX)
    else:
        return os.path.join(settings.DATASET_DIR_NAME, _STORAGE_PREFIX)

def get_detect_dir(full_path = False):
    if full_path:
        return os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, _STORAGE_PREFIX)
    else:
        return os.path.join(settings.DETECT_DIR_NAME, _STORAGE_PREFIX)

def get_train_path():
    return os.path.join(settings.TRAIN_ROOT, _STORAGE_PREFIX)

def get_model_path():
    return os.path.join(settings.MODEL_ROOT, _STORAGE_PREFIX)

def get_train_pid(train_action):
    train_dir = os.path.join(get_train_path(), str(train_action.pk))
    train_ps = os.popen('ps -ef | grep train.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if train_ps != '':
        return int(train_ps.split()[1])
    else:
        return 0

def get_eval_pid(train_action):
    train_dir = os.path.join(get_train_path(), str(train_action.pk))
    eval_ps = os.popen('ps -ef | grep eval2.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if eval_ps != '':
        return int(eval_ps.split()[1])
    else:
        return 0

def stop_train_ps(train_action):
    train_dir = os.path.join(get_train_path(), str(train_action.pk))
    train_ps = os.popen('ps -ef | grep train.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if train_ps != '':
        pid = int(train_ps.split()[1])
        os.system('kill -s 9 {}'.format(str(pid)))
    eval_ps = os.popen('ps -ef | grep eval2.py | grep {} | grep -v grep'.format(train_dir)).readline()
    if eval_ps != '':
        pid = int(eval_ps.split()[1])
        os.system('kill -s 9 {}'.format(str(pid)))
