import os
import tensorflow as tf
import logging
from nets import nets_factory
import time

logger = logging.getLogger()
logger.setLevel('INFO')

time0 = time.time()

model_dir = '/home/src/goodsdl/model'
traintype_modeldir = os.path.join(model_dir, str(67))
checkpoint = tf.train.latest_checkpoint(traintype_modeldir)
logging.info('begin loading step3 model: {}-{}'.format(7, checkpoint))

network_fn = nets_factory.get_network_fn(
    'nasnet_mobile',
    num_classes=2,
    is_training=False)
image_size = network_fn.default_image_size

time1 = time.time()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

_graph = tf.Graph()
with _graph.as_default():
    input_image_path = tf.placeholder(dtype=tf.string, name='input_image')
    image_string = tf.read_file(input_image_path)
    image = tf.image.decode_jpeg(image_string, channels=3, name='image_tensor')
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    images = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)
    logits, _ = network_fn(images)
    probabilities = tf.nn.softmax(logits, name='detection_classes')
    time2 = time.time()
    variables_to_restore = tf.global_variables()
    saver = tf.train.Saver(variables_to_restore)
    session = tf.Session(config=config)
    saver.restore(session, checkpoint)
    time3 = time.time()
    logger.info('end loading: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
