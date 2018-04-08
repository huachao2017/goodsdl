import os
import time, datetime
import logging
from urllib import request, parse
import tensorflow as tf
from dl.step20.cluster import ClusterSettings
from dl import common
import GPUtil as GPU
from dl.util import get_host_ip
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import TrainTask

tf.app.flags.DEFINE_string(
    'domain', None, 'The train server domain.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/src/goodsdl/media/dataset', 'The dataset dir.')

tf.app.flags.DEFINE_string(
    'dir_serial', '', 'dir serial string point to project')

tf.app.flags.DEFINE_integer(
    'train_interval_secs', 60, 'train interval secs.')

FLAGS = tf.app.flags.FLAGS


def _run_train(domain, serial, task):
    train_data = parse.urlencode([
        ('action', 'T30'),
        ('serial', serial),
        ('dataset_dir', task.dataset_dir),
        ('model_name', task.model_name),
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
        else:
            # 更新状态
            task.state = 1
            task.save()

    return True

def main(_):
    logger = logging.getLogger()
    logger.setLevel('INFO')

    train_interval_secs = FLAGS.train_interval_secs

    serial_postfix = '' if FLAGS.dir_serial == '' else '_'+FLAGS.dir_serial

    dataset_dir = FLAGS.dataset_dir
    domain = FLAGS.domain
    if domain is None:
        domain = get_host_ip()

    logging.info('crontab start at {} '.format(domain))


    while True:
        start = time.time()
        # FOR avoid "MySQL server has gone away"
        logging.info('Starting monitor at ' + time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.localtime()))
        tasks = TrainTask.objects.filter(state=0).order_by('-create_time')[:1]
        if len(tasks) == 0:
            logging.info('Finished all train!')
            break

        # TODO 需要远程查询所有空闲gpu
        gpus = GPU.getAvailable(order='memory', limit=1)
        if len(gpus) > 0:
            tasks = TrainTask.objects.filter(state=0).order_by('-create_time')[:1]
            if len(tasks) == 0:
                logging.info('Finished all train!')
                break

            logging.info('Starting train--{} at '.format(str(tasks[0].dataset_dir)) + time.strftime(
                '%Y-%m-%d-%H:%M:%S', time.localtime()))
            if not _run_train(domain, FLAGS.dir_serial, tasks[0]):
                break
        time_to_next_eval = start + train_interval_secs - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)

if __name__ == '__main__':
    tf.app.run()
