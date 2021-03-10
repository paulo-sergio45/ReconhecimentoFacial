import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from Main.ConexaoDB import selectFaceDB
from Main.ModeloResnet import Modelo


class Reconhecimento():
    def __init__(self, neigh, nca):
        super(Reconhecimento, self).__init__()
        self.neigh = neigh
        self.nca = nca
        self.modelo = Modelo().get_modelNeigh()

    def reconheceFace(self, feature, path1, id1, path2, id2, dis):

        img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        img1 = np.asarray(img1, dtype=np.float32) / 255.0
        img1 = np.expand_dims(img1, 0)

        feature1 = self.nca.transform(self.modelo.predict(img1))

        img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        img2 = np.asarray(img2, dtype=np.float32) / 255.0
        img2 = np.expand_dims(img2, 0)

        feature2 = self.nca.transform(self.modelo.predict(img2))

        cosi1 = cosine_distances(feature, feature1)
        simi1 = cosine_similarity(feature, feature1)
        cosi2 = cosine_distances(feature, feature2)
        simi2 = cosine_similarity(feature, feature2)

        if (simi1 > simi2 and cosi1 < cosi2):
            id = id1
            simi = simi1
            cosi = cosi1
            dist = dis[0][0]
        else:
            id = id2
            simi = simi2
            cosi = cosi2
            dist = dis[0][1]

        if (round(float(simi), 2) > 0.7 and int(dist) <= 12 and round(float(cosi) - 0.005, 2) <= 0.25):
            return id
        else:
            return "face nao reconhecida"

    def validaFace(self, img):
        feature = self.nca.transform(self.modelo.predict(img))
        dis, posi = self.neigh.kneighbors(feature, 2, return_distance=True)

        try:
            id1, path1, nome1, tel1, per1, atv, ide1, log, bai, uf, cep = selectFaceDB(int(posi[0][0] + 1))[0]
            id2, path2, nome2, tel2, per2, atv, ide2, log, bai, uf, cep = selectFaceDB(int(posi[0][1] + 1))[0]

            id = self.reconheceFace(feature, path1, id1, path2, id2, dis)
        except:
            return "face nao reconhecida"

        return id
