import argparse
import copy
import os.path as osp
import torch
import numpy as np
import random

from torch_geometric.datasets import Planetoid
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import torch_geometric.transforms as T


seed = 111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/Planetoid')
unseen_classes = [1, 2, 3]

transform = T.Compose([
    T.NormalizeFeatures(),
    T.SVDFeatureReduction(200),
    T.GDC(),
])

dataset = Planetoid(path, 'Cora', transform=transform)
data = dataset[0]

zs_data = T.RemoveTrainingClasses(unseen_classes)(copy.copy(data))

h = zs_data.x

reg = LogisticRegression()
reg.fit(h[data.train_mask].numpy(), data.y[data.train_mask].numpy())

test_acc = reg.score(h[data.test_mask].numpy(), data.y[data.test_mask].numpy())
print(f'Test Acc: {test_acc:.4f}')