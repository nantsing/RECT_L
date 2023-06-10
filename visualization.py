import numpy as np
import matplotlib.pyplot as plt

loss_paths = ['LabelUsageLoss.npy', 'LabelAsInputLoss.npy', 'RECTLoss.npy', 'ResConnectLoss.npy']

for loss_path in loss_paths:
    loss = np.load(loss_path)
    plt.plot(loss, label=loss_path[:-8])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Cora")
plt.savefig('loss.png')    
