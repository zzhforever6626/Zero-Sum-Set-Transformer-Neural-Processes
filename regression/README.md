### 1D Regression

### Training
The config of hyperparameters of each model is saved in configs/gp. If training for the first time, evaluation data will be generated and saved in evalsets/gp. Model weights and logs are saved in results/gp/{model}/{expid}.
```
python gp.py --mode=train --expid=default-zstnp --model=zstnp
```

### Evaluation
Run BO using a trained model.
```
python gp.py --mode=eval --expid=default-zstnp --model=zstnp
```

## CelebA Image Completion

### Training
Download [img_align_celeba.zip](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) and unzip. Download [list_eval_partitions.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE) and [identity_CelebA.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs). Place downloaded files in `datasets/celeba` folder. Run `python data/celeba.py` to preprocess the data.
```
python celeba.py --mode=train --expid=default-zstnp --model=zstnp
```
### Evaluation
```
python celeba.py --mode=eval --expid=default-zstnp --model=zstnp
```
If evaluating for the first time, evaluation data will be generated and saved in evalsets/celeba.

## EMNIST Image Completion

### Training
```
python emnist.py --mode=train --expid=default-zstnp --model=zstnp
```
If training for the first time, EMNIST training data will automatically downloaded and saved in `datasets/emnist`.

### Evaluation
```
python emnist.py --mode=eval --expid=default-zstnp --model=zstnp
```
If evaluating for the first time, evaluation data will be generated and saved in `evalsets/emnist`.
