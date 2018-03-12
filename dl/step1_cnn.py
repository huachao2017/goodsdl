import tensorflow as tf
import os
import numpy as np
import logging

logger = logging.getLogger("detect")


class Step1CNN:
    def __init__(self, step1_model_dir):
        self.model_dir = step1_model_dir
        self.model_path = os.path.join(self.model_dir, 'frozen_inference_graph.pb')
        self._graph = None
        self._session = None
        self._isload = False

    def load(self, config):
        logger.info('begin loading step1 model: {}'.format(self.model_path))
        self._graph = tf.Graph()
        with self._graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        logger.info('end loading step1 graph...')

        # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU50%的显存
        self._session = tf.Session(graph=self._graph, config=config)

        # Definite input and output Tensors for detection_graph
        self._image_tensor = self._graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self._detection_boxes = self._graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self._detection_scores = self._graph.get_tensor_by_name('detection_scores:0')
        self._isload = True

    def is_load(self):
        return self._isload

    def detect(self,image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores) = self._session.run(
            [self._detection_boxes, self._detection_scores],
            feed_dict={self._image_tensor: image_np_expanded})

        return (boxes, scores)

