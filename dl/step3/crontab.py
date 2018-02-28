import os
import time
import logging
from urllib import request, parse
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'domain', '192.168.1.173', 'The train server domain.')

tf.app.flags.DEFINE_integer(
    'begin_traintype', 1, 'The begin of the train type.')

tf.app.flags.DEFINE_integer(
    'end_traintype', 32, 'The end of the train type.')

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
    req.add_header('Origin', 'https://{}/api'.format(domain))
    req.add_header('User-Agent',
                   'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
    with request.urlopen(req, data=train_data.encode('utf-8')) as f:
        print('Status:', f.status, f.reason)
        if f.status != 200:
            return False

    return True

def main(_):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    train_interval_secs = FLAGS.train_interval_secs
    end_traintype = FLAGS.end_traintype
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

        if cur_traintype > end_traintype:
            logging.info('Finished all train!')
            break
        time_to_next_eval = start + train_interval_secs - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)

if __name__ == '__main__':
    tf.app.run()
