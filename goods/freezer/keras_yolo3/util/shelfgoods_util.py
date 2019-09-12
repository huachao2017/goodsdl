
from set_config import config
import time
def get_post_data_goodstopn(new_image_paths):
    if len(new_image_paths)<1:
        return None,None
    api = config.http_apis['shelf_good_cluster_topn']
    img_local_files = ''
    for img_path in new_image_paths:
        img_local_files += img_path+","
    trace_id = str(time.time())
    post_data = {
        "trace_id":trace_id,
        "img_local_files":img_local_files[:-1]
    }
    return api,post_data

def get_post_data_addgood(upc,image_path,goods_shelfgoods_id):
    trace_id = str(time.time())
    post_data = {
        "trace_id": trace_id,
        "img_local_file":image_path,
        "good_upc":upc,
        "goods_shelfgoods_id":goods_shelfgoods_id
    }
    api = config.http_apis['shelf_good_add_good']
    return api,post_data

def get_post_data_deletegood(goods_shelfgoods_id):
    trace_id = str(time.time())
    post_data = {
        "trace_id": trace_id,
        "goods_shelfgoods_id": goods_shelfgoods_id,
    }
    api = config.http_apis['shelf_good_delete_good']
    return api, post_data


if __name__=='__main__':
    new_image_paths=["/home/ai/1.jpg","/home/ai/2.jpg","/home/ai/3.jpg"]
    api, post_data = get_post_data_goodstopn(new_image_paths)
    print (post_data)