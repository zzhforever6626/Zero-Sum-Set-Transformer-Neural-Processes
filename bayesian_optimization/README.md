## 1-dimensional BO

### Training
Training is exactly the same to meta regression.
```
python 1d_gp.py --mode=train --model=zstnp --expid=default
```

### Evaluation
Run BO using a trained model.
```
python 1d_bo.py --bo_mode models --bo_kernel rbf --model zstnp --expid=default
```

## 2-dimensional BO

### Training
First, generate the training dataset, and then train.
```
python highdim_gp.py --mode=generate --model=zstnp --min_num_points=30 --max_num_points=128
```
```
python highdim_gp.py --mode=train --model=zstnp --min_num_points=30 --max_num_points=128
```

### Evaluation

Run `highdim_bo.py`.   
Please choose objective function to evaluate. The following functions are supported: `ackley`, `cosine`, `rastrigin`, `dropwave`, `goldsteinprice`, `michalewicz`, `hartmann`.

```
python highdim_bo.py --objective=dropwave --dimension=2 --model=zstnp --train_min_num_points=30 --train_max_num_points=128
```
