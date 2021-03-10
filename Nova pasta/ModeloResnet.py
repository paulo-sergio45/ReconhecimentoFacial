import os

import tensorflow as tf

class Modelo():
    def __init__(self):
        super(Modelo, self).__init__()
        self.imgAltura = 100
        self.imgLargura = 100
        self.epochs = 5
        self.batch = 35

    def get_modelSoftmax(self, classes):

        modelo = tf.keras.applications.ResNet50(weights=None,
                                                include_top=True,
                                                input_shape=(self.imgAltura, self.imgLargura, 3),
                                                pooling=max,
                                                classes=classes)
        # cria o modelo resnet50 usando a api keras

        modelo.compile(optimizer="adam",
                       loss="categorical_crossentropy",
                       metrics=['accuracy'])
        # compila o modelo base

        return modelo  # retorna o modelo

    def get_modelNeigh(self):

        if (os.path.exists('Modelos/modelNeigh.h5')):
            modeloNeigh = tf.keras.models.load_model('Modelos/modelNeigh.h5')  # carrega o modelo treinando
        else:
            modelo = tf.keras.models.load_model('Modelos/modelSoftmax.h5')  # carrega o modelo treinando

            modeloNeigh = tf.keras.applications.ResNet50(weights=None,
                                                         include_top=False,
                                                         input_shape=(self.imgAltura, self.imgLargura, 3),
                                                         pooling='avg')

            # aplica configuracoes no modelo para Extração de recursos
            modeloNeigh.set_weights(modelo.get_weights()[:318])
            #  retira acamada softmax

        modeloNeigh.compile(optimizer="adam",
                            loss="categorical_crossentropy",
                            metrics=['accuracy'])
        # compila o modelo base

        modeloNeigh.save('Modelos/modelNeigh.h5')  # salva modelo treinado

        return modeloNeigh  # retorna o modelo

    def get_batch(self):
        return self.batch

    def get_epochs(self):
        return self.epochs
