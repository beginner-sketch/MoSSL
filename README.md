# [IJCAI 2024] Multi-Modality Spatio-Temporal Forecasting via Self-Supervised Learning

**[IJCAI24] Jiewen Deng, Renhe Jiang, Jiaqi Zhang, Xuan Song, "Multi-Modality Spatio-Temporal Forecasting via Self-Supervised Learning", IJCAI, 2024.**

**Our research has been accepted for presentation at the main track of IJCAI 2024.**

**This implementation showcases our MoSSL model.** 

![image](https://github.com/beginner-sketch/MoSSL/blob/main/img/framework.png)


## Preprint Link
[![arXiv](https://img.shields.io/badge/arXiv-MoSSL-B31B1B.svg)](https://arxiv.org/abs/2405.03255)

## Citation
**Citation details will be updated once the official proceedings for IJCAI 2024 are available online.**

```
@inproceedings{ijcai2024p223,
  title     = {Multi-Modality Spatio-Temporal Forecasting via Self-Supervised Learning},
  author    = {Deng, Jiewen and Jiang, Renhe and Zhang, Jiaqi and Song, Xuan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {2018--2026},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/223},
  url       = {https://doi.org/10.24963/ijcai.2024/223},
}
```

## Multi-Modality Spatio-Temporal Dataset
**NYC Traffic Demand dataset<sup id="a1">[[1]](#f1)</sup> is collected from the New York City, which consists of 98 nodes and four transportation modalities: Bike Inflow, Bike Outflow, Taxi Inflow, and Taxi Outflow. The timespan is from April 1st, 2016 to June 30th, 2016, and the time interval is set to half an hour.**

**BJ Air Quality dataset<sup id="a2">[[2]](#f2)</sup> is collected from the Beijing Municipal Environmental Monitoring Center, which contains 10 nodes and three pollutant modalities: $PM_{2.5}$, $PM_{10}$, and $SO_2$. The timespan is from March 1st, 2013 to February 28th, 2017, and the time interval is set to one hour.**

<span id="f1">1. [^](#a1)</span> https://ride.citibikenyc.com/system-data; https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

<span id="f2">2. [^](#a2)</span> https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data

## Installation Dependencies
``` 
Python 3 (>= 3.6; Anaconda Distribution)

PyTorch (>= 1.6.0) 

Numpy >= 1.17.4

Pandas >= 1.0.3
```

## Model Training and Testing
``` 
python traintest_MoSSL.py cuda_name
```
