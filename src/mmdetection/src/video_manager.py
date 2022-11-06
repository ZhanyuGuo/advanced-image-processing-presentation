import mmcv
import numpy
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# import matplotlib
# matplotlib.use("TkAgg")


def imgHandler(img, result, score_thr=0.8):
    # img -> gry, make its memory contiguous to be faster
    gry = mmcv.bgr2gray(img)
    gry = mmcv.gray2bgr(gry)
    img = np.ascontiguousarray(img)
    gry = np.ascontiguousarray(gry)

    # get bbox and segm
    bbox_result, segm_result = result
    bboxes = np.vstack(bbox_result)
    segms = mmcv.concat_list(segm_result)
    segms = np.stack(segms, axis=0)

    # get valid data
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    segms = segms[inds, ...]

    # preserve the color of mask regions
    for i in range(segms.shape[0]):
        mask = segms[i].astype(bool)
        gry[mask] = img[mask]

    return gry


config = "configs/mmdet_homework_cfg2.py"
checkpoint = "checkpoints/epoch_12.pth"
device = "cuda"
model = init_detector(config, checkpoint, device=device)

# video = mmcv.VideoReader('video/test_video.mp4')
# img = video.read()
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.8)

video = mmcv.VideoReader("demo/test_video.mp4")
i = 0
while True:
    img = video.read()
    if type(img) != numpy.ndarray:
        break

    result = inference_detector(model, img)
    rlt = imgHandler(img, result)
    mmcv.imwrite(rlt, "demo/results/%06d.jpg" % i)
    print(str(i) + f"/{video.frame_cnt}")
    i += 1

mmcv.frames2video("demo/results", "demo/result.mp4", fourcc="mp4v")
