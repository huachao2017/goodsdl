# freezer
nohup python3 /home/source/models/research/object_detection/model_main.py \
    --pipeline_config_path=/root/train/ssd_fpn_train.config \
    --model_dir=/home/src/goodsdl/train/ssd/freezer > /root/train_ssd.out 2>&1 &

# export
python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /root/train/ssd_fpn_train.config \
    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory /home/src/goodsdl/train/ssd/freezer
# shelf