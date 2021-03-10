import numpy as np
from sklearn.metrics import accuracy_score

label = []
cont = 0
for x in range(1, 201):
    if (x != 100 and x != 19 and x != 77 and x != 182):
        cont = cont + 1
        for y in range(1, 15):
            label.append(cont)
            print(cont)

tes = np.asarray(label)

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
res = np.load("Modelos/tes.npy")
np.load = np_load_old

print(res)
print(tes)

print(accuracy_score(res, tes))
