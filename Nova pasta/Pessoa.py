import os

import cv2
import numpy as np
from joblib import dump
from sklearn.neighbors import NearestNeighbors

from Main.ConexaoDB import insertDB, updateDB, deletDB, listaFaceDB

neigh = NearestNeighbors(n_neighbors=2, radius=0.3)


def cadastrarPessoa(img, feature, nome, tel, per, log, bai, uf, cep):
    try:
        features = np.load("Modelos/features.npy")
        id = len(features) + 1
        features = np.append(features, feature, axis=0)
        dir = 'face/' + str(id)
        if (not os.path.exists(dir)):
            os.mkdir(dir)
            cv2.imwrite(dir + "/" + str(id) + ".jpg", img)
        insertDB(dir + "/" + str(id) + ".jpg", id, nome, tel, per, log, bai, uf, cep)
        neigh.fit(features)
        np.save("Modelos/features.npy", features)
        dump(neigh, 'Modelos/modeloneigh.joblib')

    except:
        features = []
        id = len(features) + 1
        features = feature
        dir = 'face/' + str(id)
        if (not os.path.exists(dir)):
            os.mkdir(dir)
            cv2.imwrite(dir + "/" + str(id) + ".jpg", img)
        insertDB(dir + "/" + str(id) + ".jpg", id, nome, tel, per, log, bai, uf, cep)

        np.save("Modelos/features.npy", features)

    return True


def alterarPessoa(id, nome, tel, per, log, bai, uf, cep):
    return updateDB(id, nome, tel, per, log, bai, uf, cep)


def deletarPessoa(id, atv):
    return deletDB(id, atv)


def listarPessoa():
    return listaFaceDB()
