import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

file_path = os.path.dirname(__file__)

# choose to use a config and initialize the detector
config_path = os.path.abspath(
    os.path.join(file_path, "../configs/mmdet_homework_cfg2.py")
)

# setup a checkpoint file to load
chkpt_path = os.path.abspath(os.path.join(file_path, "../checkpoints/epoch_12.pth"))

device = "cuda"
# device = 'cpu'

# initialize the detector
model = init_detector(config_path, chkpt_path, device=device)

# use the detector to do inference
img_path = os.path.abspath(
    os.path.join(file_path, "../balloon_dataset/balloon/val/4838031651_3e7b5ea5c7_b.jpg")
)
out_path = os.path.abspath(os.path.join(file_path, "../assets/balloon.png"))

result = inference_detector(model, img_path)

# show image
show_result_pyplot(model, img_path, result)
show_result_pyplot(model, img_path, result, out_file=out_path)
