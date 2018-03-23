import os
import tensorflow as tf
from nets import nets_factory
import time
from dl.util import get_labels_to_names
import GPUtil as GPU

import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()
from goods.models import ExportAction

def load_one(config, model_dir, export):
    time0 = time.time()

    traintype_modeldir = os.path.join(model_dir, str(export.pk))
    checkpoint = tf.train.latest_checkpoint(traintype_modeldir)
    tf.logging.info('begin loading step3 model: {}-{}'.format(export.train_action.traintype, checkpoint))
    labels_to_names = get_labels_to_names(os.path.join(traintype_modeldir, 'labels.txt'))

    network_fn = nets_factory.get_network_fn(
        export.model_name,
        num_classes=len(labels_to_names),
        is_training=False)
    image_size = network_fn.default_image_size

    time1 = time.time()

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

        tf.logging.info('end loading: %.2f, %.2f, %.2f, %.2f' % (time3 - time0, time1 - time0, time2 - time1, time3 - time2))
        return session

def load_all():
    model_dir = '/home/src/goodsdl/dl/model'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    traintype_to_session = {}
    export3s = ExportAction.objects.filter(train_action__action='T3').filter(checkpoint_prefix__gt=0).order_by(
        '-update_time')

    for export in export3s:
        traintype = export.train_action.traintype
        if traintype not in traintype_to_session:
            session = load_one(config, model_dir, export)
            traintype_to_session[traintype] = session

    GPU.showUtilization()

def main(_):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.logging.set_verbosity('INFO')
    load_all()

if __name__ == '__main__':
    tf.app.run()