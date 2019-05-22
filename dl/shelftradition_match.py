from django.conf import settings
from tradition.matcher.matcher import Matcher
from goods.models import ShelfGoods
import os
import logging
from dl import common
import cv2
import numpy as np

logger = logging.getLogger("django")

class ShelfTraditionMatch:
    def __init__(self, shopid=''):
        self._isload = False
        self._shopid = shopid
        self.added_sample_upcs_to_cnt = {}
        self._matcher = Matcher(visual=True)

    def load(self):
        logger.info('begin loading TraditionMatch')
        base_samples = ShelfGoods.objects.filter(shopid=self._shopid).filter(score2=1)
        sample_dir = ''
        for sample in base_samples:
            if sample_dir == '':
                sample_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf_sample',
                                          '{}'.format(sample.shopid),
                                          '{}'.format(sample.shelfid))
            sample_image_path = os.path.join(sample_dir, '{}_{}.jpg'.format(sample.upc, sample.pk))
            if os.path.isfile(sample_image_path):
                self._matcher.add_baseline_image(sample_image_path, sample.upc) # 基准样本不能被删除，所以直接调用_matcher的方法
            else:
                ShelfGoods.objects.get(pk=sample.pk).delete()

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

    def detect_one_with_path(self,image_path):
        upc, score = self._matcher.match_image_best_one(image_path)
        return upc, score

    def detect_one_with_cv2array(self,visual_image_path,image):
        upc, score = self._matcher.match_image_best_one_with_cv2array(visual_image_path, cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
        return upc, score

    def verify_score(self,image_path,within_upcs):
        upc, score = self._matcher.match_image_best_one(image_path,within_upcs=within_upcs)
        return score