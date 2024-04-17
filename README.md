# Multi-modality Self-Supervised Learning for Spatial-Temporal Forecasting (MoSSL)
This is a implementation of our MoSSL model.

# Installation Dependencies

Python 3 (>= 3.6; Anaconda Distribution)

PyTorch (>= 1.6.0) 

Numpy >= 1.17.4

Pandas >= 1.0.3

torch-summary (>= 1.4.5)

# MoST Dataset
NYC Traffic Demand dataset<sup id="a1">[[1]](#f1)</sup> is collected from the New York City, which consists of 98 nodes and four transportation modalities: Bike Inflow, Bike Outflow, Taxi Inflow, and Taxi Outflow. The timespan is from April 1st, 2016 to June 30th, 2016, and the time interval is set to half an hour.

BJ Air Quality dataset<sup id="a2">[[2]](#f2)</sup> is collected from the Beijing Municipal Environmental Monitoring Center, which contains 10 nodes and three pollutant modalities: PM2.5, PM10, and SO$_2$. The timespan is from March 1st, 2013 to February 28th, 2017, and the time interval is set to one hour.

<span id="f1">1. [^](#a1)</span> https://ride.citibikenyc.com/system-data; https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

<span id="f2">2. [^](#a2)</span> https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data

# Model Training
``` python
python traintest_MoSSL.py cuda_name
```
