import os
from django.conf import settings
from goods2 import common

def _add_train_image(client, upcs, one_upc_cnt = 10):
    # 上传2类图片各10张
    dataset_root_path = os.path.join(settings.MEDIA_ROOT, 'dataset', 'step2')

    for upc in upcs:
        upc_path = os.path.join(dataset_root_path, upc)
        index = 0
        for filename in os.listdir(upc_path):
            image_path = os.path.join(upc_path, filename)
            with open(image_path, mode='rb') as fp:
                response = client.post('/api2/trainimage/', {'deviceid': '1000', 'upc': upc, 'source': fp},
                                       format='multipart')

            index += 1
            if index >= one_upc_cnt:
                break


def _add_image(client, deviceid, identify, add_ground_truth=True):
    dataset_root_path = os.path.join(settings.MEDIA_ROOT, 'dataset', 'step2')
    upcs = ['4711931005106', '4714221811227']

    for upc in upcs:
        upc_path = os.path.join(dataset_root_path, upc)
        index = 0
        for filename in os.listdir(upc_path):
            image_path = os.path.join(upc_path, filename)
            with open(image_path, mode='rb') as fp:
                response = client.post('/api2/image/',
                                            {'deviceid': deviceid, 'identify': upc + identify, 'source': fp},
                                            format='multipart')

            index += 1
            if index >= 10:
                break

        if add_ground_truth:
            client.post('/api2/imagegroundtruth/', {'deviceid': deviceid, 'identify': upc + identify, 'upc': upc})

