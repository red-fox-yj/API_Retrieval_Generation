# 意图拆解&评估
环境配置
```
conda create -n IntentSplit python=3.10
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transfomers
conda install openai
```

```
# 预处理数据
python data_process.py
# 意图拆解
python intent-split.py
# 多意图检索
python muti-intent-retrieval.py
```