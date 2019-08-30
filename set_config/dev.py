#########################################YOLOV3##################################################################
yolov3_params={
    'good_model_path' :'/home/ai/model/freezer/ep3356-loss42.196-val_loss45.709.h5',
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

######################################common#####################################################################
common_params={
    'freezer_check_yolov3_switch':True
}