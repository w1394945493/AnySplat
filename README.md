# AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views

### [Project Page](https://city-super.github.io/anysplat/) | [Paper](https://arxiv.org/pdf/2505.23716)

[Lihan Jiang*](https://jianglh-whu.github.io/), [Yucheng Mao*](https://myc634.github.io/yuchengmao/), [Linning Xu](https://eveneveno.github.io/lnxu),
[Tao Lu](https://inspirelt.github.io/), [Kerui Ren](https://github.com/tongji-rkr), [Yichen Jin](), [Xudong Xu](https://scholar.google.com.hk/citations?user=D8VMkA8AAAAJ&hl=en), [Mulin Yu](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Jiangmiao Pang](https://oceanpang.github.io/), [Feng Zhao](https://scholar.google.co.uk/citations?user=r6CvuOUAAAAJ&hl=en), [Dahua Lin](http://dahua.site/), [Bo Dai<sup>†</sup>](https://daibo.info/) <br />

## Overview
<p align="center">
<img src="assets/pipeline.jpg" width=100% height=100% 
class="center">
</p>
Starting from a set of uncalibrated images, a transformer-based geometry encoder is followed by three decoder heads: $F_G$, $F_D$, and $F_C$, which respectively predict the Gaussian parameters ($\boldsymbol{\mu}$, $\sigma$, $\boldsymbol{r}$, $\boldsymbol{s}$, $\boldsymbol{c}$), the depth map $D$, and the camera poses $p$. These outputs are used to construct a set of pixel-wise 3D Gaussians, which is then voxelized into pre-voxel 3D Gaussians with the proposed Differentiable Voxelization module. From the voxelized 3D Gaussians, multi-view images and depth maps are subsequently rendered. The rendered images are supervised using an RGB loss against the ground truth image, while the rendered depth maps, along with the decoded depth $D$ and camera poses $p$, are used to compute geometry losses. The geometries are supervised by pseudo-geometry priors ($\tilde{D}$, $\tilde{p}$) obtained by the pretrained VGGT.


### The code and checkpoints are coming soon~ 🙂‍↕️