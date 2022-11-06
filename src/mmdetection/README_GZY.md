# README - GZY

## INSTALL

1. Create conda environment.
    ```bash
    conda create -n openmmlab python=3.7
    conda activate openmmlab
    ```

2. Install [pytorch](pytorch.org/).

3. Install openmmlab.
    ```bash
    pip install openmim
    mim install mmcv-full
    # git clone https://github.com/open-mmlab/mmdetection.git
    # git clone https://gitee.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -r requirements/build.txt
    pip install -v -e .
    python setup.py install
    ```

## Inference Demo

1. Download the model.
    ```bash
    cd mmdetection
    mkdir checkpoints/
    cd checkpoints/
    wget https://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295.pth
    ```

2. Run the script.
    ```bash
    cd mmdetection
    python src/image_demo.py
    ```

3. Then there will be a window and the image will be saved in `demo/`.

## Balloon Dataset

1. Download the pre-trained model.
    ```bash
    cd checkpoints/
    wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
    ```

2. Train.
    ```bash
    python src/train_balloon.py
    ```
    The model will be saved in `checkpoint/`.

3. Inference.
    ```bash
    python src/image_demo_balloon.py
    ```
    The result will be saved in `demo/`.

4. Make video.
    ```bash
    python src/video_manager.py
    ```
    The result will be saved in `demo/`.
