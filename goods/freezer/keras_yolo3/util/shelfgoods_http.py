from goods.freezer.keras_yolo3.util import shelfgoods_util
from goods.freezer.keras_yolo3.util import http_util
import logging
logger = logging.getLogger("detect")

def post_goodgetn(new_image_paths):
    # 调用批量获取图片的相似topn_upc的接口， 获取within_upc
    api, post_data = shelfgoods_util.get_post_data_goodstopn(new_image_paths)
    reponse_data = None
    if api == None:
        logging.error("good_cluster_topn , new_image_paths is null , check!!!" + str(new_image_paths))
        return None
    else:
        result = http_util.http_post(api, post_data)
        if result == None or result == '':
            logging.error("good_cluster_topn , http failed , check!!!")
            return None
        else:
            reponse_data = http_util.parse_reponse_dict(result)
    if reponse_data == None:
        logging.error("good_cluster_topn , http parse failed , check!!!")
        return None
    return reponse_data

def post_addgood(upc,image_path):
    # 调用批量获取图片的相似topn_upc的接口， 获取within_upc
    api, post_data = shelfgoods_util.get_post_data_addgood(upc,image_path)
    reponse_data = None
    if api == None:
        logging.error("addgood , check!!!upc=%s , image_path=%s" %( str(upc),str(image_path)))
        return None
    else:
        result = http_util.http_post(api, post_data)
        if result == None or result == '':
            logging.error("addgood , http failed , check!!!")
        else:
            reponse_data = http_util.parse_reponse_none(result)
    if reponse_data == None:
        logging.error("addgood , http parse failed , check!!!")
        return None
    return reponse_data

def post_deletegood(upc,image_path):
    # 调用批量获取图片的相似topn_upc的接口， 获取within_upc
    api, post_data = shelfgoods_util.get_post_data_addgood(upc,image_path)
    reponse_data = None
    if api == None:
        logging.error("deletegood , check!!!upc=%s , image_path=%s" %( str(upc),str(image_path)))
        return None
    else:
        result = http_util.http_post(api, post_data)
        if result == None or result == '':
            logging.error("deletegood , http failed , check!!!")
        else:
            reponse_data = http_util.parse_reponse_none(result)
    if reponse_data == None:
        logging.error("deletegood , http parse failed , check!!!")
        return None
    return reponse_data