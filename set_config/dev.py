#########################################YOLOV3-freezer##################################################################
yolov3_params={
    'good_model_path' :'/home/ai/model/freezer/ep3408-loss42.201-val_loss42.072.h5',
    'anchors_path' :'./goods/freezer/keras_yolo3/model_data/yolo_anchors.txt',
    'classes_path' : './goods/freezer/keras_yolo3/model_data/voc_classes.txt',
    'label_path':'./goods/freezer/keras_yolo3/model_data/goods_label_map.pbtxt',
    'score' :0.25,
    'iou' :0.45,
    'model_image_size' : (416, 416),
    'gpu_num' : 1,
    "diff_switch_iou":(True,0.6),
    "single_switch_iou_minscore":(True,0.0,0.3)
}

#########################################YOLOV3-shelfgood##################################################################
yolov3_shelf_good_params={
    'good_model_path' :'/home/ai/model/shelfgood/ep34645-loss81.243-val_loss111.489.h5',
    'anchors_path' :'./goods/freezer/keras_yolo3/model_data/shelfgoods_yolo_anchors.txt',
    'classes_path' : './goods/freezer/keras_yolo3/model_data/shelfgoods_voc_classes.txt',
    'label_path':'./goods/freezer/keras_yolo3/model_data/shelfgoods_label_map.pbtxt',
    'score' :0.05,
    'iou' :0.45,
    'model_image_size' : (416, 416),
    'gpu_num' : 1,
}

######################################http-api#####################################################################
http_apis={
    "shelf_good_cluster_topn":"http://127.0.0.1:8000/api/get_goods_topn",
    "shelf_good_add_good":"http://127.0.0.1:8000/api/add_new_good",
    "shelf_good_delete_good": "http://127.0.0.1:8000/api/delete_good",
}
######################################common#####################################################################
common_params={
    'freezer_check_yolov3_switch':True, # 冰柜yolo检测
    'shelfgoods_check_yolov3_switch': False,# 陈列yolo检测
    'shelfgoods_check_cluster_switch':True,#陈列聚类检测
    'shelfgoods_check_match_switch': True,  # 陈列匹配检测
}