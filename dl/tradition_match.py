from tradition.matcher.matcher import Matcher
from goods.models import SampleImageClass

class TraditionMatch:
    def __init(self, deviceid=''):
        self._isload = False
        self._deviceid = deviceid
        self._matcher = Matcher()

    def load(self):
        samples = SampleImageClass.objects.filter(deviceid=self._deviceid)
        for sample in samples:
            self._matcher.add_baseline_image(sample.source.path)

        self._isload = True

    def is_load(self):
        return self._isload

    def detect(self,image_path):
        key, cnt = self._matcher.match_image_best_one(image_path)

        # TODO need test top_n
        # top_5 = self._matcher.match_image_top_n(image_path)
        return key
