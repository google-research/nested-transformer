# [Nested Hierarchical Transformer](https://arxiv.org/pdf/2105.12723.pdf) Official Jax Implementation

NesT is a simple method, which aggregates nested local transformers on image blocks. The idea makes vision transformers attain better accuracy, data efficiency, and convergence on the ImageNet benchmark. NesT can be scaled to small datasets to match convnet accuracy.

This is not an officially supported Google product.


## Pretrained Models and Results


|   Model     |  Accuracy   |  Checkpoint path       |
|------------:|------------:|-----------------------:|
|   Nest-B    |  **83.8**   | [gs://gresearch/nest-checkpoints/nest-b_imagenet](https://console.cloud.google.com/storage/browser/gresearch/nest-checkpoints/nest-b_imagenet)
|   Nest-S    |    83.3     | [gs://gresearch/nest-checkpoints/nest-s_imagenet](https://console.cloud.google.com/storage/browser/gresearch/nest-checkpoints/nest-s_imagenet)
|   Nest-T    |    81.5     | [gs://gresearch/nest-checkpoints/nest-t_imagenet](https://console.cloud.google.com/storage/browser/gresearch/nest-checkpoints/nest-t_imagenet)

Note: Accuracy is evaluated on the ImageNet2012 validation set.


#### Tensorbord.dev

See ImageNet training [logs](https://tensorboard.dev/experiment/AU4DxhjnRBieaPsgCWGxng/#scalars) at Tensorboard.dev.


## Colab

Colab is available for test: https://colab.sandbox.google.com/github/google-research/nested-transformer/blob/main/colab.ipynb


## Pytorch re-implementation
The timm library has incorporated [NesT and pre-trained models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nest.py) in Pytorch. 


## Instruction on Image Classification

### Environment setup

```bash
virtualenv -p python3 --system-site-packages nestenv
source nestenv/bin/activate

pip install -r requirements.txt
```

### Evaluate on ImageNet

At the first time, download ImageNet following `tensorflow_datasets` instruction
from command lines. Optionally, download all pre-trained checkpoints

```bash
bash ./checkpoints/download_checkpoints.sh
```

Run the evaluation script to evaluate NesT-B.

```bash
python main.py --config configs/imagenet_nest.py --config.eval_only=True \
  --config.init_checkpoint="./checkpoints/nest-b_imagenet/ckpt.39" \
  --workdir="./checkpoints/nest-t_imagenet_eval"
```

### Train on ImageNet
The default configuration trains NesT-B on TPUv2 8x8 with per device batch size 16.

```bash
python main.py --config configs/imagenet_nest.py --jax_backend_target=<TPU_IP_ADDRESS> --jax_xla_backend="tpu_driver" --workdir="./checkpoints/nest-b_imagenet"
```

Note: See [jax/cloud_tpu_colab](https://github.com/google/jax/blob/master/cloud_tpu_colabs/README.md) for info about TPU_IP_ADDRESS.



#### Train NesT-T on 8 GPUs.

```bash
python main.py --config configs/imagenet_nest_tiny.py --workdir="./checkpoints/nest-t_imagenet_8gpu"
```

The codebase does not support multi-node GPU training (>8 GPUs). The models reported in our
paper is trained using TPU with 1024 total batch size.


### Train on CIFAR

```bash
# Recommend to train on 2 GPUs. Training NesT-T can use 1 GPU.
CUDA_VISIBLE_DEVICES=0,1 python  main.py --config configs/cifar_nest.py --workdir="./checkpoints/nest_cifar"
```

## Cite

```
@inproceedings{zhang2021aggregating,
  title={Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding},
  author={Zizhao Zhang and Han Zhang and Long Zhao and Ting Chen and and Sercan Ö. Arık and Tomas Pfister},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}
```

