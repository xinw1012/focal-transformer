# Focal Transformer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focal-self-attention-for-local-global/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=focal-self-attention-for-local-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focal-self-attention-for-local-global/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=focal-self-attention-for-local-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focal-self-attention-for-local-global/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=focal-self-attention-for-local-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focal-self-attention-for-local-global/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=focal-self-attention-for-local-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focal-self-attention-for-local-global/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=focal-self-attention-for-local-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/focal-self-attention-for-local-global/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=focal-self-attention-for-local-global)

This is the official implementation of our [Focal Transformer -- "Focal Self-attention for Local-Global Interactions in Vision Transformers"](https://arxiv.org/pdf/2107.00641.pdf), 
by Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan and Jianfeng Gao.

## Introduction

![focal-transformer-teaser](figures/focal-transformer-teaser.png)

Our Focal Transfomer introduced a new self-attention mechanism called **focal self-attention** for vision transformers. 
In this new mechanism, **each token attends the closest surrounding tokens at fine granularity but the tokens far away at coarse granularity**, 
and thus can capture both short- and long-range visual dependencies efficiently and effectively. 

With our Focal Transformers, we achieved superior performance over the state-of-the-art vision Transformers on a range of public benchmarks. 
In particular, our Focal Transformer models with a moderate size of 51.1M and a larger size of 89.8M achieve `83.6 and 84.0` Top-1 accuracy, respectively, 
on ImageNet classification at 224x224 resolution. 
Using Focal Transformers as the backbones, we obtain consistent and substantial improvements over the current state-of-the-art methods 
for 6 different object detection methods trained with standard 1x and 3x schedules. 
Our largest Focal Transformer yields `58.7/58.9 box mAPs` and `50.9/51.3 mask mAPs` on COCO mini-val/test-dev, 
and `55.4 mIoU` on ADE20K for semantic segmentation.

## Benchmarking 

### Image Classification on [ImageNet-1K](https://www.image-net.org/)

| Model | Pretrain | Resolution | acc@1 | acc@5 | #params | FLOPs | Checkpoint | Config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Focal-T | IN-1K | 224 | 82.2 | 95.9 | 28.9M   | 4.9G   | [download](https://projects4jw.blob.core.windows.net/model/focal-transformer/imagenet1k/focal-tiny-is224-ws7.pth) | [yaml](configs/focal_tiny_patch4_window7_224.yaml) |
| Focal-S | IN-1K | 224 | 83.6 | 96.2 | 51.1M   | 9.4G   | [download](https://projects4jw.blob.core.windows.net/model/focal-transformer/imagenet1k/focal-small-is224-ws7.pth) |[yaml](configs/focal_small_patch4_window7_224.yaml) |
| Focal-B | IN-1K | 224 | 84.0 | 96.5 | 89.8M   | 16.4G  | [download](https://projects4jw.blob.core.windows.net/model/focal-transformer/imagenet1k/focal-base-is224-ws7.pth) | [yaml](configs/focal_base_patch4_window7_224.yaml) |

### Object Detection and Instance Segmentation on [COCO](https://cocodataset.org/#home)

#### [Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)

| Backbone | Pretrain | Lr Schd | #params | FLOPs | box mAP | mask mAP | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Focal-T | ImageNet-1K | 1x | 49M | 291G | 44.8 | 41.0 | 
| Focal-T | ImageNet-1K | 3x | 49M | 291G | 47.2 | 42.7 | 
| Focal-S | ImageNet-1K | 1x | 71M | 401G | 47.4 | 42.8 | 
| Focal-S | ImageNet-1K | 3x | 71M | 401G | 48.8 | 43.8 | 
| Focal-B | ImageNet-1K | 1x | 110M | 533G | 47.8 | 43.2 | 
| Focal-B | ImageNet-1K | 3x | 110M | 533G | 49.0 | 43.7 | 

#### [RetinaNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)

| Backbone | Pretrain | Lr Schd | #params | FLOPs | box mAP | 
| :---: | :---: | :---: | :---: | :---: | :---: |
| Focal-T | ImageNet-1K | 1x | 39M | 265G | 43.7 |
| Focal-T | ImageNet-1K | 3x | 39M | 265G | 45.5 | 
| Focal-S | ImageNet-1K | 1x | 62M | 367G | 45.6 | 
| Focal-S | ImageNet-1K | 3x | 62M | 367G | 47.3 | 
| Focal-B | ImageNet-1K | 1x | 101M | 514G | 46.3 | 
| Focal-B | ImageNet-1K | 3x | 101M | 514G | 46.9 | 

#### Other detection methods

| Backbone | Pretrain | Method | Lr Schd | #params | FLOPs | box mAP | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Focal-T | ImageNet-1K | [Cascade Mask R-CNN](https://arxiv.org/abs/1712.00726) | 3x | 87M  | 770G | 51.5 | 
| Focal-T | ImageNet-1K | [ATSS](https://arxiv.org/pdf/1912.02424.pdf)           | 3x | 37M  | 239G | 49.5 |
| Focal-T | ImageNet-1K | [RepPointsV2](https://arxiv.org/pdf/2007.08508.pdf)    | 3x | 45M  | 491G | 51.2 | 
| Focal-T | ImageNet-1K | [Sparse R-CNN](https://arxiv.org/pdf/2011.12450.pdf)   | 3x | 111M | 196G | 49.0 | 

### Semantic Segmentation on [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

| Backbone | Pretrain  | Method | Resolution | Iters | #params | FLOPs | mIoU | mIoU (MS) | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Focal-T | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 62M  | 998G | 45.8 | 47.0 | 
| Focal-S | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 85M | 1130G | 48.0 | 50.0 | 
| Focal-B | ImageNet-1K  | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 512x512 | 160k | 126M | 1354G | 49.0 | 50.5 | 
| Focal-L | ImageNet-22K | [UPerNet](https://arxiv.org/pdf/1807.10221.pdf) | 640x640 | 160k | 240M | 3376G | 54.0 | 55.4 | 

## Getting Started

* Please follow [get_started_for_image_classification.md](./classification/get_started.md) to get started for image classification.
* Please follow [get_started_for_object_detection.md](./detection/get_started.md) to get started for object detection.
* Please follow [get_started_for_semantic_segmentation.md](./segmentation/get_started.md) to get started for semantic segmentation.

## Citation

If you find this repo useful to your project, please consider to cite it with following bib:

    @misc{yang2021focal,
        title={Focal Self-attention for Local-Global Interactions in Vision Transformers}, 
        author={Jianwei Yang and Chunyuan Li and Pengchuan Zhang and Xiyang Dai and Bin Xiao and Lu Yuan and Jianfeng Gao},
        year={2021},
        eprint={2107.00641},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

## Acknowledgement

Our codebase is built based on [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We thank the authors for the nicely organized code!

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
