# 【Review Under at The Visual Computer】Enhancing Industrial Defect Detection: A Nonsymmetric Skip Fusion Autoencoder Approach
This repository contains the code and dataset for the paper titled "Enhancing Industrial Defect Detection: A Nonsymmetric Skip Fusion Autoencoder Approach", currently under review at The Visual Computer. 
##Setup

### Mvtec AD Dataset  

For Mvtec evaluation code install:

```
numpy==1.18.5
Pillow==7.0.0
scipy==1.7.1
tabulate==0.8.7
tifffile==2021.7.30
tqdm==4.56.0
```

Download dataset (if you already have downloaded then set path to dataset (`--mvtec_ad_path`) when calling `efficientad.py`).

```
mkdir mvtec_anomaly_detection
cd mvtec_anomaly_detection
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
cd ..
```

Download evaluation code:

```
wget https://www.mydrive.ch/shares/60736/698155e0e6d0467c4ff6203b16a31dc9/download/439517473-1665667812/mvtec_ad_evaluation.tar.xz
tar -xvf mvtec_ad_evaluation.tar.xz
rm mvtec_ad_evaluation.tar.xz
```
