import numpy as np


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MOE.router_knn import KNNRouter

Z = np.load("embedings/Output embeddings/all_train_Z.npy")
y = np.load("embedings/Output embeddings/all_train_expert_y.npy")

router = KNNRouter(
    "embedings/Output embeddings/all_train_Z.npy",
    "embedings/Output embeddings/all_train_expert_y.npy",
    k=5
)

# prueba con embeddings del mismo train
pred = router.predict(Z[:100])

print("Accuracy router:", (pred == y[:100]).mean())