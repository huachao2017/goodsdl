pip install -r requirements.txt
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_24_10_2017.tar.gz
tar -xvf faster_rcnn_nas_coco_24_10_2017.tar.gz -C /home/src/dl
mv /home/src/dl/faster_rcnn_nas_coco_24_10_2017/model.ckpt.* /home/src/dl