import json

import matplotlib.pyplot as plt
import numpy as np

f = open("Modelos/trainHistorico1.json", )
data1 = json.load(f)

loss = data1['loss']
val_loss = data1['val_loss']
accuracy = data1['accuracy']
val_accuracy = data1['val_accuracy']

f = open("Modelos/trainHistorico2.json", )
data2 = json.load(f)

loss = np.append(loss, data2['loss'])
val_loss = np.append(val_loss, data2['val_loss'])
accuracy = np.append(accuracy, data2['accuracy'])
val_accuracy = np.append(val_accuracy, data2['val_accuracy'])

f = open("Modelos/trainHistorico3.json", )
data3 = json.load(f)

loss = np.append(loss, data3['loss'])
val_loss = np.append(val_loss, data3['val_loss'])
accuracy = np.append(accuracy, data3['accuracy'])
val_accuracy = np.append(val_accuracy, data3['val_accuracy'])

f = open("Modelos/trainHistorico4.json", )
data4 = json.load(f)

loss = np.append(loss, data4['loss'])
val_loss = np.append(val_loss, data4['val_loss'])
accuracy = np.append(accuracy, data4['accuracy'])
val_accuracy = np.append(val_accuracy, data4['val_accuracy'])

f = open("Modelos/trainHistorico5.json", )
data5 = json.load(f)

loss = np.append(loss, data5['loss'])
val_loss = np.append(val_loss, data5['val_loss'])
accuracy = np.append(accuracy, data5['accuracy'])
val_accuracy = np.append(val_accuracy, data5['val_accuracy'])

f = open("Modelos/trainHistorico6.json", )
data6 = json.load(f)

loss = np.append(loss, data6['loss'])
val_loss = np.append(val_loss, data6['val_loss'])
accuracy = np.append(accuracy, data6['accuracy'])
val_accuracy = np.append(val_accuracy, data6['val_accuracy'])

x = {'loss': loss,
     'val_loss': val_loss,
     'accuracy': accuracy,
     'val_accuracy': val_accuracy
     }

# summarize history for accuracy
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('precisão do modelo')
plt.ylabel('precisão')
plt.xlabel('época')
plt.legend(['treino', 'teste'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('perda do modelo')
plt.ylabel('perda')
plt.xlabel('época')
plt.legend(['treino', 'teste'], loc='upper left')
plt.show()
