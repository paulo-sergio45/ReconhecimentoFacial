import cv2
import numpy as np
from joblib import load
from mtcnn import MTCNN

from Main.ConexaoDB import selectFaceDB
from Main.Normalizacao import alinhaFace, normalizaFace
from Main.ReconhecimentoFace import Reconhecimento

detector = MTCNN()
neigh = load('Modelos/modeloneigh.joblib')  # ler modelo ja treinado
nca = load('Modelos/modelonca.joblib')
rec = Reconhecimento(neigh, nca)


def ReconheceFaceImagem(paht):
    try:
        imagem = cv2.imread(paht, cv2.IMREAD_COLOR)
        detections = detector.detect_faces(imagem)
    except:
        print("imagem nao reconhecida")
    else:
        for detection in detections:
            score = detection["confidence"]
            if score >= 0.95:
                keypoints = detection["keypoints"]
                x, y, w, h = detection['box']
                new_image = alinhaFace(normalizaFace(imagem), keypoints)

                detected_face = new_image[y:y + h, x - ((x + h) - (w + x)):x + h]

                img0 = cv2.resize(detected_face, (100, 100), interpolation=cv2.INTER_AREA)

                img = np.asarray(img0, dtype=np.float32) / 255.0
                img = np.expand_dims(img, 0)
                pess = rec.validaFace(img)
                if (pess != "face nao reconhecida"):
                    id, path, nome, tel, per, atv, ide2, log, bai, uf, cep = selectFaceDB(int(pess))[0]

                    print("Id:{0} \nNome:{1} \nTelefone:{2} \nPermissao:{3}\n"
                          "Logradouro:{4} \nBairro:{5} \nUF:{6} \nCEP:{4}".format(
                        str(id), str(nome), str(tel), str(per), str(log), str(bai), str(uf), str(cep)))

                    text = "Id:{0}  Nome:{1}".format(str(id), str(nome))
                    cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(imagem, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow("oi", imagem)
                    cv2.waitKey(0)
                else:
                    text = "face nao reconhecida"
                    cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(imagem, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow("oi", imagem)
                    cv2.waitKey(0)

            else:
                print("face nao detectada")


def main():
    # paht = "baseFEI/1-02.jpg"
    paht = "imagens/teste.jpg"
    ReconheceFaceImagem(paht)


if __name__ == "__main__":
    main()
