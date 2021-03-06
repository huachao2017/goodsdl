import os
import time, datetime
import logging
from urllib import request, parse
import tensorflow as tf
from dl.step20.cluster import ClusterSettings
from dl import common
import GPUtil as GPU
from dl.util import get_host_ip, get_labels_to_names
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import ExportAction

tf.app.flags.DEFINE_string(
    'domain', None, 'The train server domain.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/src/goodsdl/media/dataset', 'The dataset dir.')

tf.app.flags.DEFINE_string(
    'dir_serial', '', 'dir serial string point to project')

tf.app.flags.DEFINE_integer(
    'train_interval_secs', 60, 'train interval secs.')

tf.app.flags.DEFINE_integer(
    'start_traintype', 0, 'start train type.')

FLAGS = tf.app.flags.FLAGS


def _run_train(domain, serial, traintype,dataset_dir):
    train_data = parse.urlencode([
        ('action', 'T21'),
        ('serial', serial),
        ('traintype', traintype),
        ('dataset_dir', dataset_dir),
        ('model_name', 'nasnet_large'),
        ('is_fineture', False),
    ])
    req = request.Request('http://{}/api/trainaction/'.format(domain))
    req.add_header('Origin', 'http://{}/api/'.format(domain))
    req.add_header('User-Agent',
                   'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
    # print(req.full_url)
    with request.urlopen(req, data=train_data.encode('utf-8')) as f:
        print('Status:', f.status, f.reason)
        if f.status != 201:
            return False

    return True

def main(_):
    logger = logging.getLogger()
    logger.setLevel('INFO')

    train_interval_secs = FLAGS.train_interval_secs

    serial_postfix = '' if FLAGS.dir_serial == '' else '_'+FLAGS.dir_serial

    dataset_dir = FLAGS.dataset_dir
    index = 1
    export20s = ExportAction.objects.filter(train_action__action='T20').order_by('-update_time')[:1]
    export20 = export20s[0]
    labels_to_names = get_labels_to_names(os.path.join('/home/src/goodsdl/dl/model', str(export20.pk), 'labels.txt'))
    print(labels_to_names)
    end_traintype = len(labels_to_names) - 1
    cur_traintype = FLAGS.start_traintype
    domain = FLAGS.domain
    if domain is None:
        domain = get_host_ip()

    logging.info('crontab start at {} '.format(domain))


    while True:
        start = time.time()
        # FOR avoid "MySQL server has gone away"
        exports = ExportAction.objects.filter(train_action__action='T21').order_by('-update_time')
        logging.info('Starting monitor at ' + time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.localtime()) + ', complete {}'.format(len(exports)))
        if cur_traintype > end_traintype:
            logging.info('Finished all train!')
            break

        # TODO 需要远程查询所有空闲gpu
        gpus = GPU.getAvailable(order='memory', limit=2)
        if len(gpus) > 1:
            # 检查目录样本是否符合
            for i in range(10):
                step21_dataset_dir = os.path.join(dataset_dir, common.STEP20_PREFIX+serial_postfix, labels_to_names[cur_traintype])
                logging.info(step21_dataset_dir)
                if not os.path.isdir(step21_dataset_dir):
                    cur_traintype += 1
                elif len(os.listdir(step21_dataset_dir)) <= 1: # 必须大于两类才能分类
                    cur_traintype += 1
                else:
                    break

            # 检查是否已经训练，并且训练后没有新增样本
            exports = ExportAction.objects.filter(train_action__action='T21').filter(train_action__traintype=cur_traintype).order_by('-update_time')[:1]
            if len(exports)>0:
                step21_dataset_dir = os.path.join(dataset_dir, common.STEP20_PREFIX+serial_postfix, labels_to_names[cur_traintype])
                filetime = datetime.datetime.fromtimestamp((os.path.getmtime(step21_dataset_dir)))
                if filetime < exports[0].update_time:
                    logging.info('Skip train--{} because has trained in '.format(str(cur_traintype)) + time.strftime('%Y-%m-%d-%H:%M:%S', exports[0].update_time.timetuple()))
                    cur_traintype += 1
                    continue

            logging.info('Starting train--{} at '.format(str(cur_traintype)) + time.strftime(
                '%Y-%m-%d-%H:%M:%S', time.localtime()))
            if not _run_train(domain, FLAGS.dir_serial, cur_traintype, os.path.join(dataset_dir, common.STEP20_PREFIX+serial_postfix,labels_to_names[cur_traintype])):
                break
            cur_traintype += 1

        if cur_traintype > end_traintype:
            logging.info('Finished all train!')
            break
        time_to_next_eval = start + train_interval_secs - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)

if __name__ == '__main__':
    tf.app.run()
