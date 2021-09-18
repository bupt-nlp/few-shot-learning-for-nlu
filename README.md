# few-shot-learning-for-slu

A Baseline for Few-Shot Leanring For NLU. 

## Features

* Common FewShotExample interface for all of dataset
* Plug and Play Models for FewShot Models
* Plug and Play Metrics Method for Few Shot Loss
* ......

## Docs

### Config

There are two config for few-shot and training period.

```python
class Config(Tap):
    dataset: FewShotDataSet    # snips
    distance: Metric      # metric
    n_way_train: int            # numbner of support examples per class for training tasks
    n_way_validation: int       # numbner of support examples per class for validation task
    
    k_shot_train: int            # number of classes for training tasks
    k_shot_validation: int       # number of classes for validation tasks
```

```python
class TrainConfig(Tap):
    epoch: int
    steps: int          # steps per epoch
    learning_rate: float
    num_task: int
```

### FewShot Examples

```python

```


## Reference Projects

* [few-shot](https://github.com/oscarknagg/few-shot)
* []()

## Changelog 


## Creators

- [@wj-Mcat](https://github.com/wj-Mcat) - Jingjing WU (吴京京)

## Copyright & License

- Code & Docs © 2021 wj-Mcat, 吴京京
- Code released under the Apache-2.0 License
- Docs released under Creative Commons
