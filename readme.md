# Generate Multi-label Adversarial Examples by Linear Programming

## Requirements
- python-3.6.6
- numpy-1.16.5
- torch-1.3.1
- torchnet-0.0.4
- torchvision-0.2.0
- tqdm-4.31.1
- mosek-9.1.7

## Usage
- apply license of the package mosek on [https://www.mosek.com/products/academic-licenses/](https://www.mosek.com/products/academic-licenses/) and move the license file 'mosek.lic' to folder /home/username/mosek/mosek.lic
- create new conda environment(the environment name is mlalp)
```
conda env create -f mlalp_conda_env.yaml
```

- download the VOC2007 and VOC2012 dataset and move to the folder 'data/voc2007/VOCdevkit/VOC2012/' or 'data/voc2012/VOCdevkit/VOC2012/'
- download ML-GCN model and ML-LIW model from <a href="#download">model download</a> or you can train the model yourself
```
cd ml_gcn_model or cd ml_liw_model
python train.py
```

- move the model to folder 'checkpoint/mlgcn' or 'checkpoint/mlliw'
- go the code folder
```
cd src
```

- run attack
```
python demo_mlgcn_voc2007.py --adv_batch_size=10 --adv_method='mla_lp' --target_type='hide_single'
python demo_mlgcn_voc2012.py --adv_batch_size=10 --adv_method='mla_lp' --target_type='hide_single'
python demo_mlliw_voc2007.py --adv_batch_size=10 --adv_method='mla_lp' --target_type='hide_single'
python demo_mlliw_voc2012.py --adv_batch_size=10 --adv_method='mla_lp' --target_type='hide_single'
```

- test attack performance

<a id="download"/>

## Download Adversarial Data and Model
[Adversarial Data and Model](https://rec.ustc.edu.cn/share/19a80830-3602-11ea-91d2-cf19be005ac6)

## Thanks
- https://github.com/Megvii-Nanjing/ML-GCN