from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

# choose to use a config and initialize the segmentor
config = "configs/fcn/fcn_r50-d8_512x1024_80k_cityscapes.py"

# setup a checkpoint file to load
checkpoint = "checkpoints/fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth"

device = "cuda"
# device='cpu'

# initialize the segmentor
model = init_segmentor(config, checkpoint, device=device)

# use the segmentor to do inference
img = "demo/demo.png"
result = inference_segmentor(model, img)

palette = "cityscapes"
opacity = 0.5

# show image
show_result_pyplot(
    model,
    img,
    result,
    get_palette(palette),
    opacity=opacity,
    out_file="demo/result.jpg",
)
