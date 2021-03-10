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


def ReconheceFaceVideo(paht):
    cap = cv2.VideoCapture(paht)

    while (cv2.waitKey(1) < 0):

        try:
            conectado, frame = cap.read()
            detections = detector.detect_faces(frame)
        except:
            print("imagem nao reconhecida")
        else:
            if not conectado:
                break

            for detection in detections:
                score = detection["confidence"]
                if score >= 0.95:
                    try:
                        keypoints = detection["keypoints"]
                        x, y, w, h = detection['box']
                        new_image = alinhaFace(normalizaFace(frame), keypoints)

                        detected_face = new_image[y:y + h, x - ((x + h) - (w + x)):x + h]

                        img0 = cv2.resize(detected_face, (100, 100), interpolation=cv2.INTER_AREA)
                    except:
                        print("erro ao normalizar imagens")
                    else:
                        img = np.asarray(img0, dtype=np.float32) / 255.0
                        img = np.expand_dims(img, 0)
                        pess = rec.validaFace(img)
                        if (pess != "face nao reconhecida"):
                            id, path, nome, tel, per, atv, ide2, log, bai, uf, cep = selectFaceDB(int(pess))[0]

                            print("Id:{0} \nNome:{1} \nTelefone:{2} \nPermissao:{3}\n"
                                  "Logradouro:{4} \nBairro:{5} \nUF:{6} \nCEP:{4}".format(
                                str(id), str(nome), str(tel), str(per), str(log), str(bai), str(uf), str(cep)))

                            img1 = cv2.imread(path, cv2.IMREAD_COLOR)
                            text = "Id:{0}  Nome:{1}".format(str(id), str(nome))

                        else:
                            text = "face nao reconhecida"

                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 50, 50), 2)
                        cv2.putText(frame, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow("oi", frame)
                        cv2.waitKey(1)

                else:
                    print("face nao detectada")
    cap.release()
    cv2.destroyAllWindows()


def main():
    paht = "Videos/teste_video.mp4"
    ReconheceFaceVideo(paht)


if __name__ == "__main__":
    main()
