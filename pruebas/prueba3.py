import numpy as np
y = np.load("embedings/Output embeddings/all_train_expert_y.npy")

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))