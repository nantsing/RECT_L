# RECT fine-tuning
## metrics
```python
# hyper-parameters
unseen_classes = [1, 2, 3]
random_seed = 42
```

| model | CiteSeer | Cora | PubMed |
|:----------:|:-------:|:------:|:----:|
| logist reg | 0.5760 | 0.6300 | 0.7340 |
| rectl | 0.6700 | 0.7270 | 0.7450 |
| rectl (w/o train)| __0.7010__ | __0.7370__ | 0.7430|
| rectl (w/ label_as_input) | 0.6920 | 0.7320 | 0.7260 |
| rectl (w/ label_as_input+usage) | 0.6800 | 0.7300 | 0.7230 |
| rectl (w/ residual connections) | 0.6780 | 0.7340 | __0.7530__ |
| rectl (w/ random feature) | 0.2960 | 0.4700 | 0.4470 |
