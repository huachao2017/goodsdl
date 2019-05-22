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
    def __init__(self, shopid, shelfid):
        self.added_sample_upcs_to_cnt = {}
        self._matcher = Matcher(visual=True)

        logger.info('begin loading TraditionMatch')
        base_samples = ShelfGoods.objects.filter(shopid=shopid).filter(shelfid=shelfid).filter(score2=1)
        sample_dir = os.path.join(settings.MEDIA_ROOT, settings.DETECT_DIR_NAME, 'shelf_sample',
                                  '{}'.format(shopid),
                                  '{}'.format(shelfid))
        for sample in base_samples:
            sample_image_path = os.path.join(sample_dir, '{}_{}.jpg'.format(sample.upc, sample.pk))
            if os.path.isfile(sample_image_path):
                self._matcher.add_baseline_image(sample_image_path, sample.upc) # 基准样本不能被删除，所以直接调用_matcher的方法
            else:
                ShelfGoods.objects.get(pk=sample.pk).delete()

        logger.info('end loading TraditionMatch:{}'.format(self._matcher.get_baseline_cnt()))

    def detect_one_with_path(self,image_path):
        upc, score = self._matcher.match_image_best_one(image_path)
        return upc, score

    def detect_one_with_cv2array(self,visual_image_path,image):
        upc, score = self._matcher.match_image_best_one_with_cv2array(visual_image_path, cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR))
        return upc, score

    def verify_score(self,image_path,within_upcs):
        upc, score = self._matcher.match_image_best_one(image_path,within_upcs=within_upcs)
        return score