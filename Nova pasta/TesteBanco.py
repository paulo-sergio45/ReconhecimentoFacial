import cv2
import numpy as np
from joblib import load
from mtcnn import MTCNN
from sklearn.neighbors import NearestNeighbors

from Main.ModeloResnet import Modelo
from Main.Normalizacao import alinhaFace, normalizaFace
from Main.Pessoa import cadastrarPessoa

detector = MTCNN()
mode = Modelo()
modelo = mode.get_modelNeigh()
# chama o modelo
neigh = NearestNeighbors(n_neighbors=2, radius=0.3)
nca = load('Modelos/modelonca.joblib')


def testeValidaFace(imagem):
    detections = detector.detect_faces(imagem)

    for detection in detections:
        score = detection["confidence"]
        if score >= 0.95:
            keypoints = detection["keypoints"]
            x, y, w, h = detection['box']
            new_image = alinhaFace(normalizaFace(imagem), keypoints)

            detected_face = new_image[y:y + h, x - ((x + h) - (x + w)):x + h]

            img0 = cv2.resize(detected_face, (100, 100), interpolation=cv2.INTER_AREA)
            img = np.asarray(img0, dtype=np.float32) / 255.0
            img = np.expand_dims(img, 0)

            feature = nca.transform(modelo.predict(img))
            try:
                features = np.load("Modelos/features.npy")
            except:
                features = []
            id = len(features) + 1
            cadastrarPessoa(img0, feature, id, id, id, id, id, id, id)


        else:
            print("face nao reconhecida")


def main():
    for x in range(1, 200):
        print(x)
        if (x == 100):
            None
        else:

            paht = "baseFEI/" + str(x) + "-11.jpg"
            imagem = cv2.imread(paht, cv2.IMREAD_COLOR)
            testeValidaFace(imagem)


if __name__ == "__main__":
    main()
