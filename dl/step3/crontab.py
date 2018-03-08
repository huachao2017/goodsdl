import os
import time
import logging
from urllib import request, parse
import tensorflow as tf
from dl.step2.cluster import ClusterSettings

tf.app.flags.DEFINE_string(
    'domain', '192.168.1.170', 'The train server domain.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/src/goodsdl/media/dataset', 'The dataset dir.')

tf.app.flags.DEFINE_integer(
    'begin_traintype', 0, 'The begin of the train type.')

tf.app.flags.DEFINE_integer(
    'end_traintype', 0, 'The end of the train type.')

tf.app.flags.DEFINE_integer(
    'train_interval_secs', 60, 'train interval secs.')

FLAGS = tf.app.flags.FLAGS


def _run_train(domain, traintype):
    train_data = parse.urlencode([
        ('action', 'T3'),
        ('traintype', traintype),
        ('is_fineture', False),
        ('desc', 'step3 -- '+str(traintype)),
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

    dataset_dir = FLAGS.dataset_dir
    cluster_settings = ClusterSettings(os.path.join(dataset_dir,'step2/cluster.txt'))
    end_traintype = FLAGS.end_traintype
    cur_traintype = FLAGS.begin_traintype
    if end_traintype == 0:
        end_traintype = cluster_settings.get_max_traintype()
    if cur_traintype == 0:
        cur_traintype = 1
    domain = FLAGS.domain
    while True:
        start = time.time()
        logging.info('Starting monitor at ' + time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.gmtime()))
        train_ps = os.popen('ps -ef | grep train.py | grep -v grep').readline()
        if train_ps == '':
            # 检查目录
            for i in range(10):
                step3_dataset_dir = os.path.join(dataset_dir, 'step3', str(cur_traintype))
                if len(os.listdir(step3_dataset_dir)) <= 1: # 必须大于两类才能分类
                    cur_traintype += 1
                else:
                    break

            logging.info('Starting train--{} at '.format(str(cur_traintype)) + time.strftime(
                '%Y-%m-%d-%H:%M:%S', time.gmtime()))
            if not _run_train(domain, cur_traintype):
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
