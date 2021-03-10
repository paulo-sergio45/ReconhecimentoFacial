import cv2
import numpy as np
from joblib import load
from mtcnn import MTCNN

from Main.ConexaoDB import selectFaceDB
from Main.ModeloResnet import Modelo
from Main.Normalizacao import alinhaFace, normalizaFace
from Main.Pessoa import cadastrarPessoa
from Main.ReconhecimentoFace import Reconhecimento


def ReconheceFaceVideo(paht, rec, detector):
    cap = cv2.VideoCapture(paht)

    while (True):

        conectado, frame = cap.read()
        if not conectado:
            break

        detections = detector.detect_faces(frame)

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
    detector = MTCNN()

    nca = load('Modelos/modelonca.joblib')
    mode = Modelo()
    modelo = mode.get_modelNeigh()
    neigh = load('Modelos/modeloneigh.joblib')  # ler modelo ja treinado
    rec = Reconhecimento(neigh, nca)

    paht = "Videos/teste_video.mp4"
    # ReconheceFaceVideo(paht, rec, detector)

    # pahti = "imagens/teste01.jpg"
    # imagem = cv2.imread(pahti, cv2.IMREAD_COLOR)
    # img = np.asarray(imagem, dtype=np.float32) / 255.0
    # img = np.expand_dims(img, 0)
    #
    # feature = nca.transform(modelo.predict(img))


    # cadastrarPessoa(imagem, feature, "paulo", "999999", "1", "rua dos bobos", "itara", "ES", "029047")

    neigh = load('Modelos/modeloneigh.joblib')  # ler modelo ja treinado
    rec = Reconhecimento(neigh, nca)
    ReconheceFaceVideo(paht, rec, detector)


if __name__ == "__main__":
    main()
