import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig


with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg

@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    from src.dataset.data_module import DataModule
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        None,
        global_rank=0,
    )
    re10k_dataset = data_module.test_dataloader()

    for data in re10k_dataset:

        print(data.keys())
    return

if __name__=='__main__':

    main()