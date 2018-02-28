import os
import time
import logging
from dl.step2 import cluster
from urllib import request, parse
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'domain', '192.168.1.173', 'The train server domain.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/src/goodsdl/media/dataset', 'The path of the dataset dir.')
tf.app.flags.DEFINE_integer(
    'begin_traintype', 1, 'The begin of the train type.')
tf.app.flags.DEFINE_integer(
    'train_interval_secs', 600, 'train interval secs.')
FLAGS = tf.app.flags.FLAGS


def _run_train(domain, traintype):
    train_data = parse.urlencode([
        ('action', 'T3'),
        ('traintype', traintype),
        ('is_fineture', False),
        ('desc', 'step3 -- '+str(traintype)),
    ])
    req = request.Request('https://{}/api/trainaction'.format(domain))
    with request.urlopen(req, data=train_data.encode('utf-8')) as f:
        print('Status:', f.status, f.reason)
        if f.status != 200:
            return False

    return True

def main(_):
    train_interval_secs = FLAGS.train_interval_secs
    step2_dir = os.path.join(FLAGS.dataset_dir, 'step2')
    cluster_filepath = os.path.join(step2_dir, 'cluster.txt')
    cluster_setting = cluster.ClusterSettings(cluster_filepath)
    max_number_of_traintype = cluster_setting.get_max_traintype()
    cur_traintype = FLAGS.begin_traintype
    domain = FLAGS.domain
    while True:
        start = time.time()
        logging.info('Starting monitor at ' + time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.gmtime()))
        train_ps = os.popen('ps -ef | grep train.py | grep -v grep').readline()
        if train_ps == '':
            logging.info('Starting train--{} at '.format(str(cur_traintype)) + time.strftime(
                '%Y-%m-%d-%H:%M:%S', time.gmtime()))
            if not _run_train(domain, cur_traintype):
                break
            cur_traintype += 1

        if cur_traintype > max_number_of_traintype:
            logging.info('Finished all train!')
            break
        time_to_next_eval = start + train_interval_secs - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)

if __name__ == '__main__':
    tf.app.run()
