import numpy as np

Z = np.load("embedings/Output embeddings/all_train_Z.npy")
y = np.load("embedings/Output embeddings/all_train_expert_y.npy")

print("Shape Z:", Z.shape)
print("Norm promedio:", np.mean(np.linalg.norm(Z, axis=1)))
print("Min norm:", np.min(np.linalg.norm(Z, axis=1)))
print("Max norm:", np.max(np.linalg.norm(Z, axis=1)))