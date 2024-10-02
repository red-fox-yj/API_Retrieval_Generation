# Pipeline
## 环境配置
```
conda create -n Pipeline python=3.10
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Quick start
```
srun -p a800 --output=pipeline.log --gres=gpu:1 --cpus-per-task=12 --mem-per-cpu=4G python pipeline.py
```