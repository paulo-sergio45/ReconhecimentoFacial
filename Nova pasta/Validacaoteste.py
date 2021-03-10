import cv2
import numpy as np
from joblib import load
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from Main.ConexaoDB import selectFaceDB
from Main.ModeloResnet import Modelo
from Main.Normalizacao import alinhaFace, normalizaFace

detector = MTCNN()
modelo = Modelo().get_modelNeigh()  # chama o modelo
neigh = load('Modelos/modeloneigh.joblib')  # ler modelo ja treinado
nca = load('Modelos/modelonca.joblib')


def validaFace(imagem):
    detections = detector.detect_faces(imagem)

    for detection in detections:
        score = detection["confidence"]
        if score >= 0.99:
            keypoints = detection["keypoints"]
            x, y, w, h = detection['box']
            new_image = alinhaFace(normalizaFace(imagem), keypoints)

            detected_face = new_image[y:y + h, x - ((x + h) - (x + w)):x + h]

            img0 = cv2.resize(detected_face, (100, 100), interpolation=cv2.INTER_AREA)
            imgc = np.asarray(img0, dtype=np.float32) / 255.0
            imgc = np.expand_dims(imgc, 0)

            feature = nca.transform(modelo.predict(imgc))

            # extrai os recursos da base de dados

            dis, posi = neigh.kneighbors(feature, 2, return_distance=True)
            # print(dis, posi + 1)
            id1 = selectFaceDB(int(posi[0][0] + 1))[0]
            id2 = selectFaceDB(int(posi[0][1] + 1))[0]
            # print(id1,id2)

            img1 = cv2.imread(id1[1], cv2.IMREAD_COLOR)
            imgc = np.asarray(img1, dtype=np.float32) / 255.0
            imgc1 = np.expand_dims(imgc, 0)

            feature1 = nca.transform(modelo.predict(imgc1))

            img2 = cv2.imread(id2[1], cv2.IMREAD_COLOR)
            imgc = np.asarray(img2, dtype=np.float32) / 255.0
            imgc1 = np.expand_dims(imgc, 0)

            feature2 = nca.transform(modelo.predict(imgc1))

            cosi1 = cosine_distances(feature, feature1)
            simi1 = cosine_similarity(feature, feature1)
            cosi2 = cosine_distances(feature, feature2)
            simi2 = cosine_similarity(feature, feature2)

            if (simi1 >= simi2 and cosi1 <= cosi2):
                id = id1[0]

            else:
                id = id2[0]

            return id
        else:
            return -1
    return -1


def main():
    p1 = []
    for x in range(1, 201):
        if (x != 100 and x != 19 and x != 77 and x != 182):
            for y in range(1, 15):
                if (y < 10):
                    y = "0" + str(y)

                paht = "baseFEI/" + str(x) + "-" + str(y) + ".jpg"
                imagem = cv2.imread(paht, cv2.IMREAD_COLOR)

                tes = validaFace(imagem)
                print(tes)

                p1.append(tes)

    p1 = np.asarray(p1)
    np.save("Modelos/tes.npy", p1)


if __name__ == "__main__":
    main()
