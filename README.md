# [DiffPose: Video Setting)
[[Paper]](https://arxiv.org/abs/2211.16940) | [[Project Page]](https://GONGJIA0208.github.io/Diffpose/) | [[SUTD-VLG Lab]](https://github.com/sutdcv)


### Environment

The code is developed and tested under the following environment:

-   Python 3.8.2
-   PyTorch 1.7.1
-   CUDA 11.0

You can create the environment via:

```bash
conda env create -f environment.yml
```

### Dataset

Our datasets are based on [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline) and [Video3D data](https://github.com/facebookresearch/VideoPose3D). We provide the GMM format data generated from the above datasets [here](https://www.dropbox.com/sh/54lwxf9zq4lfzss/AABmpOzg31PrhxzcxmFQt3cYa?dl=0). You should put the downloaded files into the `./data` directory.
Note that we only change the format of the Video3D data to make them compatible with our GMM-based DiffPose training strategy, and the value of the 2D pose in our dataset is the same as them.

## Video-based experiments
### Evaluating pre-trained models for frame-based experiments

We provide the pre-trained diffusion model (with CPN-dected 2D Pose as input) [here](https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?dl=0). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_video.py \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/mixste_cpn_243f.bin \
--model_diff_path checkpoints/diffpose_video_uvxyz_cpn.pth \
--doc t_human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_cpn.out 2>&1 &
```

We also provide the pre-trained diffusion model (with Ground truth 2D pose as input) [here](https://www.dropbox.com/sh/jhwz3ypyxtyrlzv/AABivC5oiiMdgPePxekzu6vga?dl=0). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_diffpose_video.py \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/mixste_cpn_243f.bin \
--model_diff_path checkpoints/diffpose_video_uvxyz_gt.pth \
--doc t_human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_gt.out 2>&1 &
```


### Bibtex

If you find our work useful in your research, please consider citing:

    @InProceedings{gong2023diffpose,
        author    = {Gong, Jia and Foo, Lin Geng and Fan, Zhipeng and Ke, Qiuhong and Rahmani, Hossein and Liu, Jun},
        title     = {DiffPose: Toward More Reliable 3D Pose Estimation},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
    }

## Acknowledgement

Part of our code is borrowed from [DDIM](https://github.com/ermongroup/ddim), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), [Graformer](https://github.com/Graformer/GraFormer), [MixSTE](https://github.com/JinluZhang1126/MixSTE) and [PoseFormer](https://github.com/zczcwh/PoseFormer). We thank the authors for releasing the codes.
