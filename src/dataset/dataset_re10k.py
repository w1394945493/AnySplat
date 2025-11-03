from io import BytesIO
import json
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor

from pathlib import Path
from typing import Literal
from dataclasses import dataclass

from torch.utils.data import Dataset


from .dataset import DatasetCfgCommon
from .view_sampler import ViewSampler,ViewSamplerCfg
from .types import Stage
from .shims.crop_shim import apply_crop_shim_re10k as apply_crop_shim




@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False

@dataclass
class DatasetRe10kCfgWrapper:
    re10k: DatasetRE10kCfg


class DatasetRE10k(Dataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        self.data_root = cfg.roots[0] / self.stage
        root_chunks = sorted(
            [path for path in self.data_root.iterdir() if path.suffix == ".torch"]
        )
        self.data_chunks = []
        for chunk in root_chunks:
            self.data_chunks.extend(torch.load(chunk))



    def __len__(self):
        return len(self.data_chunks)
    def __getitem__(self, index_tuple):
        index = index_tuple[0]
        example = self.data_chunks[index]
        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        scene = example["key"]
        context_indices, target_indices = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
        )
        # Load the images.
        context_images = [
            example["images"][index.item()] for index in context_indices
        ]
        context_images = self.convert_images(context_images)
        target_images = [
            example["images"][index.item()] for index in target_indices
        ]
        target_images = self.convert_images(target_images)

        nf_scale = 1.0
        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / nf_scale,
                "far": self.get_bound("far", len(context_indices)) / nf_scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
            },
            "scene": scene,
        }
        example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))

        return example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)