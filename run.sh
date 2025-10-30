
CUDA_VISIBLE_DEVICES=5 python -m src.main \
    +experiment=re10k \
    mode=test \
    dataset.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/re10k/re10k_subset] \
    checkpointing.load=/home/lianghao/wangyushen/data/wangyushen/Weights/anysplat/model.safetensors \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/any_splat/test \

