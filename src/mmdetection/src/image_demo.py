from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# choose to use a config and initialize the detector
config = "configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py"

# setup a checkpoint file to load
checkpoint = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295.pth"

device = "cuda"
# device='cpu'

# initialize the detector
model = init_detector(config, checkpoint, device=device)

# use the detector to do inference
img = "demo/demo.jpg"
result = inference_detector(model, img)

# show image
show_result_pyplot(model, img, result)
show_result_pyplot(model, img, result, out_file="demo/result.jpg")
