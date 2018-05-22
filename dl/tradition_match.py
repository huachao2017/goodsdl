from tradition.matcher.matcher import Matcher
from goods.models import SampleImageClass
import os
import logging
from dl import common

logger = logging.getLogger("detect")

class TraditionMatch:
    def __init__(self, deviceid='', step=common.STEP2_PREFIX):
        self._isload = False
        self.step = step
        self._deviceid = deviceid
        self.added_sample_upcs_to_cnt = {}
        self._matcher = Matcher(visual=True)

    def load(self):
        logger.info('begin loading TraditionMatch')
        if self.step == common.STEP2_PREFIX:
            base_samples = SampleImageClass.objects.filter(deviceid='')
        elif self.step == common.STEP2S_PREFIX:
            base_samples = SampleImageClass.objects.filter(deviceid=common.STEP2S_PREFIX)
        else:
            base_samples = []
        for sample in base_samples:
            if os.path.isfile(sample.source.path):
                self._matcher.add_baseline_image(sample.source.path, sample.upc) # 基准样本不能被删除，所以直接调用_matcher的方法
            else:
                SampleImageClass.objects.get(pk=sample.pk).delete()

        samples = SampleImageClass.objects.filter(deviceid=self._deviceid)
        for sample in samples:
            self.add_baseline_image(sample.source.path, sample.upc) # 新增样本可以被删除，调用类中的方法

        logger.info('end loading TraditionMatch:{}'.format(self._matcher.get_baseline_cnt()))
        self._isload = True

    def is_load(self):
        return self._isload

    def add_baseline_image(self, image_path, upc):
        if os.path.isfile(image_path):
            self._matcher.add_baseline_image(image_path, upc)
            if upc in self.added_sample_upcs_to_cnt:
                self.added_sample_upcs_to_cnt[upc] += 1
            else:
                self.added_sample_upcs_to_cnt[upc] = 1
            logger.info('baseline image({}): {}'.format(self._matcher.get_baseline_cnt(), image_path))
        else:
            logger.error('baseline image not exist: {}'.format(image_path))

    def removeall_baseline_image(self):
        for upc in self.added_sample_upcs_to_cnt:
            self._matcher.removeall_baseline_image(upc)

    def detect(self,image_paths):

        upcs = []
        scores = []
        for image_path in image_paths:
            upc, score = self._matcher.match_image_best_one(image_path)
            upcs.append(upc)
            scores.append(score)

        return upcs, scores

    def detect_one(self,image_path):
        upc, score = self._matcher.match_image_best_one(image_path)
        return upc, score

    def verify_score(self,image_path,within_upcs):
        upc, score = self._matcher.match_image_best_one(image_path,within_upcs=within_upcs)
        return score