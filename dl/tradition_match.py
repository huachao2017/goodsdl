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
                self._matcher.add_baseline_image(sample.source.path, sample.upc)
            else:
                SampleImageClass.objects.get(pk=sample.pk).delete()

        logger.info('end loading TraditionMatch')
        self._isload = True

    def is_load(self):
        return self._isload

    def add_baseline_image(self, image_path, upc):
        if os.path.isfile(image_path):
            self._matcher.add_baseline_image(image_path, upc)

    def detect(self,image_paths):

        upcs = []
        scores = []
        for image_path in image_paths:
            upc, score = self._matcher.match_image_best_one(image_path)
            upcs.append(upc)
            scores.append(score)

        return upcs, scores

    def verify_score(self,image_path,upc):
        upc, score = self._matcher.match_image_best_one(image_path,within_upcs=[upc])
        return score