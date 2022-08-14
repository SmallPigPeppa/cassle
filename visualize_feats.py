# from tsne_torch import TorchTSNE as TSNE
# import torch
# from cassle.methods.simclr import SimCLR
# model=
#
# pretrained_model_path="/home/admin/code/cassle_v129.0/experiments/2022_08_12_23_59_47-simclr-cifar100-contrastive/exggc94o/simclr-cifar100-contrastive-task0-ep=499-exggc94o.ckpt"
# print(f"Loading previous task checkpoint {pretrained_model_path}...")
# state_dict = torch.load(pretrained_model_path, map_location="cpu")["state_dict"]
# model.load_state_dict(state_dict, strict=False)
#
# X = ...  # shape (n_samples, d)
# X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(X)  # returns shape (n_samples, 2)

# tsne on classes [10,11,..,20]

# visualize


import matplotlib.pyplot as plt
import numpy as np

# Getting unique labels

x1=np.random.rand(40,2)
x2=x1+2
x3=x1+3

# plotting the results:

# for i in u_labels:
plt.scatter(x1[:,0], x1[:,1], label='class-1')
plt.scatter(x3[:,0], x3[:,1], label='class-1')
plt.scatter(x2[:,0], x2[:,1], label='class-2')
plt.legend()
# plt.show()
plt.savefig('/Users/lwz/Desktop/result.pdf')