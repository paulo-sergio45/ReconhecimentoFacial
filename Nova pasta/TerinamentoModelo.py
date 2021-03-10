import json
import os.path

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from Main.ModeloResnet import Modelo


class Treinamento():

    def __init__(self):
        super(Treinamento, self).__init__()

    def treinamentoSoftmax(self, data, labels):

        arquivo_modelo = 'Modelos/melhor_modeloSoftmax.h5'
        checkpointer = ModelCheckpoint(arquivo_modelo, monitor='val_loss', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        le = LabelEncoder()
        lab = le.fit_transform(labels)
        onehot = OneHotEncoder(sparse=False)
        lab = lab.reshape(len(lab), 1)
        one = onehot.fit_transform(lab)
        # converte rótulos de várias classes

        (trainX, testX, trainY, testY) = train_test_split(data,
                                                          one, test_size=0.25, random_state=45)

        if (os.path.exists('Modelos/modelSoftmax.h5')):
            modelo = tf.keras.models.load_model('Modelos/modelSoftmax.h5')  # ler modelo ja treinado
        else:
            modelo = Modelo.get_modelSoftmax(len(le.classes_))
            # chama o modelo

        historico = modelo.fit(x=trainX,
                               y=trainY,
                               validation_split=0.25,
                               batch_size=Modelo.get_batch,
                               epochs=Modelo.get_epochs,
                               shuffle=True,
                               callbacks=[checkpointer, reduce_lr]
                               )
        # treina o modelo

        modelo.save('Modelos/modelSoftmax.h5')  # salva modelo treinado
        json.dump(historico, open("Modelos/trainHistorico.json", 'w'))  # salva historico de treinamento

        return historico  # retorna o historico
