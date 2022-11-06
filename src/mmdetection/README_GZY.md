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
## Inference Deomo
1. Download the model.
    ```bash
    # currently in mmdetection/
    mkdir checkpoints/
    cd checkpoints/
    wget https://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295.pth
    ```
2. Run the script.
    ```bash
    # currently in mmdetection/
    python src/image_demo.py
    ```
3. Then there will be a window and the image will be saved in `demo/`.