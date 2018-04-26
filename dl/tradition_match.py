from tradition.matcher.matcher import Matcher
from goods.models import SampleImageClass
import os
import logging

logger = logging.getLogger("detect")

class TraditionMatch:
    def __init__(self, deviceid=''):
        self._isload = False
        self._deviceid = deviceid
        self._matcher = Matcher()

    def load(self):
        logger.info('begin loading TraditionMatch')
        samples = SampleImageClass.objects.filter(deviceid=self._deviceid)
        for sample in samples:
            if os.path.isfile(sample.source.path):
                self._matcher.add_baseline_image(sample.source.path)

        logger.info('end loading TraditionMatch')
        self._isload = True

    def is_load(self):
        return self._isload

    def add_baseline_image(self, image_path):
        if os.path.isfile(image_path):
            self._matcher.add_baseline_image(image_path)

    def detect(self,image_paths):

        upcs = []
        scores = []
        for image_path in image_paths:
            key, cnt = self._matcher.match_image_best_one(image_path)
            upcs.append(key)
            if cnt < 10:
                scores.append(cnt/10)
            else:
                scores.append(0.99)


        # TODO need test top_n
        # top_5 = self._matcher.match_image_top_n(image_path)
        return upcs, scores
