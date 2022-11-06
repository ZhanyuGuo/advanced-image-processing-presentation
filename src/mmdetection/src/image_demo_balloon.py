from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# import matplotlib
# matplotlib.use("TkAgg")

config = "configs/mmdet_homework_cfg2.py"

checkpoint = "checkpoints/epoch_12.pth"

device = "cuda"
# device = 'cpu'

model = init_detector(config, checkpoint, device=device)

img = "balloon_dataset/balloon/val/4838031651_3e7b5ea5c7_b.jpg"

result = inference_detector(model, img)

show_result_pyplot(model, img, result)
# show_result_pyplot(model, img, result, score_thr=0.8)
model.show_result(img, result, out_file="demo/balloon.jpg")
