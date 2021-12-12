# Motion Prediction
Human motion prediction is the problem of forecasting future body poses given observed pose sequence. The problem requires encoding both spatial and temporal aspects of human motion, and generating sequences conditioned on that. The problem is formulated as a sequence modeling task -- the input is 120 poses (frames) i.e., 2s of motion at 60Hz, and the output is 24 poses (400ms).

## Set Up Instructions
```sh
conda create -n motion_prediction python=3.6
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/facebookresearch/fairmotion.git --no-deps
```

## Data
We use the CMU dataset which contains ~10 hours of recorded human motion data. Dataset is contained within the large (~40GB) synthetic dataset found [here](dip.is.tue.mpg.de).

## Metrics
Mean (Euler) Angle Error of poses at 80, 160, 320 and 400 ms. Mean Angle Error is the Euclidean distance between predicted and reference Euler angles averaged over all joints, summed up to N frames. See `metrics.euler_diff` for code that computes this metric.

## Preprocessing
Ensure that `data/raw` contains `CMU` directory.

```sh
cd scripts/
python make_dataset.py
```

## Training
```sh
cd scripts/
python train.py --architecture transformer --epochs 100
```
