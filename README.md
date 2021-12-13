# Motion Prediction
Human motion prediction is the problem of forecasting future body poses given observed pose sequence. The problem requires encoding both spatial and temporal aspects of human motion, and generating sequences conditioned on that. The problem is formulated as a sequence modeling task -- the input is 120 poses (frames) i.e., 2s of motion at 60Hz, and the output is 24 poses (400ms).

## Set Up Instructions
```sh
conda create -n motion_prediction python=3.6
conda activate motion_prediction
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric-temporal
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/fairmotion.git --no-deps
```

## Data
We use the CMU dataset which contains ~10 hours of recorded human motion data. Dataset is contained within the large (~40GB) synthetic dataset found [here](dip.is.tue.mpg.de).

## Metrics
Mean (Euler) Angle Error of poses at 80, 160, 320 and 400 ms. Mean Angle Error is the Euclidean distance between predicted and reference Euler angles averaged over all joints, summed up to N frames. See `metrics.euler_diff` for code that computes this metric.

## Preprocessing
Ensure that `data/raw` contains `CMU` directory.

```sh
conda activate motion_prediction
cd scripts/
python make_dataset.py
```

## Training
Currently supported architectures:
  - seq2seq
  - transformer
  - transformer_encoder

```sh
conda activate motion_prediction
cd scripts/
python train.py --architecture <architecture> --epochs 100
```
