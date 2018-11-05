from django.conf import settings
import os

TRAIN_STATE_WAITING = 1                 # 准备训练
TRAIN_STATE_TRAINING = 5                # 正在训练
TRAIN_STATE_STOP = 9                    # 正在中断
TRAIN_STATE_COMPLETE = 10               # 训练完成
TRAIN_STATE_COMPLETE_WITH_STOP = 20     # 中断完成
TRAIN_STATE_COMPLETE_WITH_ERROR = 30    # 错误完成

TASK_STATE_DOING = 1
TASK_STATE_COMPLETE = 10
TASK_STATE_ERROR = 20

DEVICE_STATE_TESTING = 0
DEVICE_STATE_COMMERCIAL = 10

# good_neighbour_bind_deviceid_list = ['2286','2995','2996','2294','2296','3135','3136','3137','3138','3139','3213']
good_neighbour_bind_deviceid_list = ['3475','3476']

_STORAGE_PREFIX = 'goods2'
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
