# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:37:00 2024

@author: Ing. Biologica
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys

sys.path.append('../lib')
import classes as cl

features = load("../data/FEATURES.joblib")
label = load("../data/label.joblib")

# Calculo la cantidad de sujetos
label = np.array(label)
cantSujetos = np.max(label[:, 2])

# Hago un sorteo a ver que sujeto va para cada conjunto
sorteo = np.random.permutation(cantSujetos) + 1

# Por ahora lo hardcodeo para 6 sujetos
indices_test = list(np.where(label[:, 2]==sorteo[0])[0])
indices_val = list(np.where(label[:, 2]==sorteo[1])[0])

indices_train = []
for j in sorteo[2:]:
    indices_train.extend(np.where(label[:, 2]==j)[0])
    
# le agrego las ventanas reposo
# Calculo los indices de label que son ventanas de reposo
ind_reposo = np.where(label[:, 0]==-1)[0]

#-------------------PARTE TODA COPIADA DEL OTRO CODIGO-------------------------
# primero sorteo para que me queden mejor distribuidas en cada conjunto
cant_features_reposo = len(ind_reposo)
sorteo = np.random.permutation(cant_features_reposo)
cant_train = (2*cant_features_reposo)//3    # aproximadamente el mismo porcentaje que se manda de cada gesto
cant_test = (168*cant_features_reposo)//1000  # lo mismo, aprox 16,8%

ind_train_reposo = ind_reposo[sorteo[:cant_train]]
ind_test_reposo = ind_reposo[sorteo[cant_train:cant_train+cant_test]]
ind_val_reposo = ind_reposo[sorteo[cant_train+cant_test:]]

# Agrego los indices de reposo a las listas que ya tenia
# Train
indices_train.extend(ind_train_reposo)

# Validacion
indices_val.extend(ind_val_reposo)

# Test
indices_test.extend(ind_test_reposo)

# lo paso a numpy array para poder definir de forma mas comoda los conjuntos de train test y val
features = np.array(features)


X_train = features[indices_train, :]
y_train = label[indices_train, 1]
X_val = features[indices_val, :]
y_val = label[indices_val, 1]
X_test =  features[indices_test, :]
y_test = label[indices_test, 1]

# Definimos a mano la cantidad de features
featureNum = 15
chNum = 16

# Numero de columnas que va a haber en features
inputDim = chNum*featureNum

# Numero de gestos contando el reposo
clasNum = np.max(label[:, 1]) + 1

# # Architecture 80 - 250 - 80
# red = cl.Clasificador(model= "mlp", arch = [80,250,80], catNum = clasNum, featureNum = inputDim)
# red.train(X_train, y_train, X_val, y_val)
# y_test_predict = red.predict(X_test)
# porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
# print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
# con_mat_test_1 = red.confMatrix(X_test, y_test)

# # Architecture 250 - 500 - 250
# red = cl.Clasificador(model= "mlp", arch = [250,500,250], catNum = clasNum, featureNum = inputDim)
# red.train(X_train, y_train, X_val, y_val)
# y_test_predict = red.predict(X_test)
# porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
# print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
# con_mat_test_2 = red.confMatrix(X_test, y_test)

# Architecture 250 - 500 - 500 - 250
red = cl.Clasificador(model= "mlp", arch = [250,500,500,250], catNum = clasNum, featureNum = inputDim)
red.train(X_train, y_train, X_val, y_val)
y_test_predict = red.predict(X_test)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
con_mat_test_3 = red.confMatrix(X_test, y_test)

# # Architecture 500 - 1000 - 500
# red = cl.Clasificador(model= "mlp", arch = [500,1000,500], catNum = clasNum, featureNum = inputDim)
# red.train(X_train, y_train, X_val, y_val)
# y_test_predict = red.predict(X_test)
# porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
# print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
# con_mat_test_4 = red.confMatrix(X_test, y_test)