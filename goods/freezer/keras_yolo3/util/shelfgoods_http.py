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
        print (api)
        print (post_data)
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

def post_addgood(upc,image_path,goods_shelfgoods_id):
    # 调用批量获取图片的相似topn_upc的接口， 获取within_upc
    api, post_data = shelfgoods_util.get_post_data_addgood(upc,image_path,goods_shelfgoods_id)
    reponse_data = None
    if api == None:
        logging.error("addgood , check!!!upc=%s , image_path=%s , goods_shelfgoods_id=%s" %( str(upc),str(image_path),str(goods_shelfgoods_id)))
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

def post_deletegood(goods_shelfgoods_id):
    # 调用批量获取图片的相似topn_upc的接口， 获取within_upc
    api, post_data = shelfgoods_util.get_post_data_deletegood(goods_shelfgoods_id)
    reponse_data = None
    if api == None:
        logging.error("deletegood , check!!! goods_shelfgoods_id=%s" %( str(goods_shelfgoods_id)))
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


if __name__=='__main__':
    new_image_paths = ['/home/src/goodsdl/media/images/shelf/4044_22/single/18_20190911_150059.jpg','/home/src/goodsdl/media/images/shelf/4044_22/single/19_20190911_150059.jpg']
    reponse_data = post_goodgetn(new_image_paths)
    print (list(reponse_data['/home/src/goodsdl/media/images/shelf/4044_22/single/18_20190911_150059.jpg']))