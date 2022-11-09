from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# choose to use a config and initialize the detector
config = "configs/mmdet_homework_cfg2.py"

# setup a checkpoint file to load
checkpoint = "checkpoints/epoch_12.pth"

device = "cuda"
# device = 'cpu'

# initialize the detector
model = init_detector(config, checkpoint, device=device)

# use the detector to do inference
img = "balloon_dataset/balloon/val/4838031651_3e7b5ea5c7_b.jpg"
result = inference_detector(model, img)

# show image
show_result_pyplot(model, img, result)
show_result_pyplot(model, img, result, out_file="demo/balloon.jpg")
