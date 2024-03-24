# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:47:20 2024

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

'''Voy a dividir los datos a mano de la siguiente forma:
Un sujeto repite cada gesto entre 5 y 6 veces. Voy a usar uno para test,
uno para validacion, y los restantes para train.
En este sentido, voy a guardar los indices de label en listas distintas, para identificar
luego que filas de FEATURES tengo que usar para cada cosa
'''

# Esto funciona porque los gestos iguales estan agrupados
# Hago un diff en el id del gesto para saber en que indices se cambia el gesto
label = np.array(label)

# Calculo los indices de label que son ventanas de reposo para trabajar con ellas por separado
ind_reposo = np.where(label[:, 0]==-1)[0]

# Calculo lo contrario ahora para facilitar obtener los labels de las ventanas con gestos
ind_gestos = np.where(label[:, 0]!=-1)[0]

label_gestos = label[ind_gestos, :]

division_gestos = np.where(np.diff(label_gestos[:, 1])!=0)[0]

# Ahora hago un diff sobre la id de ventana para saber cuando cambia. Esto me va a servir para poder separarlo
# en train, val y test
division_id_ventanas = np.where(np.diff(label_gestos[:, 0])!=0)[0]

# Aca voy a guardar el id de la ventana
train = []
val = []
test = []

for i in range(len(division_gestos)):  # 
    tope = division_gestos[i]   # este es limite con el proximo gesto
    
    if i==0:
        inicio = 0
        ind_inicio_ventana = 0
        
    else:
        inicio = division_gestos[i-1]    
        ind_inicio_ventana = np.where(division_id_ventanas==inicio)[0][0]

    # busco en division_id_ventana cuantos elementos hay entre inicio y tope
    ind_tope_ventana = np.where(division_id_ventanas==tope)[0][0]
    
    # cantidad de repeticiones en el gesto actual (del sujeto actual)
    cant_ventanas = ind_tope_ventana - ind_inicio_ventana
    
    if i != 0:
        # sorteo las ventanas que van a cada grupo
        sorteo = np.random.permutation(cant_ventanas)
        sorteo += 1     # correccion de rango
    else:
        sorteo = np.random.permutation(cant_ventanas+1)
    # El primero va a test, el segundo a val y el resto a train
    test.append(ind_inicio_ventana + sorteo[0])
    val.append(ind_inicio_ventana + sorteo[1])
    for j in range(2, len(sorteo)):
        train.append(ind_inicio_ventana + sorteo[j])

#%%
# Quiero visualizar con diferentes colores las ventanas que van al train, val y test
# ventanas_train = []
# ventanas_val = []
# ventanas_test = []

# plt.figure()
# plt.plot(label[:, 1]/np.max(label[:, 1]), label = 'gesto')
# plt.plot(label[:, 0]/np.max(label[:, 0]), label = 'ventana')
# for i in range(len(train)): # 
#     ventana_act_train = np.where(label[:, 0]==train[i])[0]
#     ventanas_train.append(ventana_act_train)
#     plt.plot(ventana_act_train, label[ventana_act_train, 0]/np.max(label[:, 0]), color = 'green')
    
#     if i < len(val):   # val y test tienen menos datos
#         ventana_act_val = np.where(label[:, 0]==val[i])[0]
#         ventanas_val.append(ventana_act_val)
#         plt.plot(ventana_act_val, label[ventana_act_val, 0]/np.max(label[:, 0]), color = 'blue')
        
#         ventana_act_test = np.where(label[:, 0]==test[i])[0]
#         ventanas_test.append(ventana_act_test)
#         plt.plot(ventana_act_test, label[ventana_act_test, 0]/np.max(label[:, 0]), color = 'black')

# plt.legend()
#%%
# Procedo a armar los conjuntos de train, val y test para las ventanas con gesto
# Tengo identificada que repeticion va para cada conjunto, ahora identifico los indices de las features usando label
indices_train = []
for j in range(len(train)):
    indices_train.extend(np.where(label[:, 0]==train[j])[0])

indices_val = []
indices_test = []
for j in range(len(val)):
    indices_val.extend(np.where(label[:, 0]==val[j])[0])
    indices_test.extend(np.where(label[:, 0]==test[j])[0])

#%% Ahora agrego las ventanas de reposo
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

#%%
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

# Architecture 80 - 250 - 80
red = cl.Clasificador(model= "mlp", arch = [80,250,80], catNum = clasNum, featureNum = inputDim)
red.train(X_train, y_train, X_val, y_val)
y_test_predict = red.predict(X_test)
porcentaje_aciertos_test = 100*np.mean(y_test_predict==y_test)
print('Porcentaje de aciertos en conjunto de TEST: %.2f' % porcentaje_aciertos_test)
con_mat_test_1 = red.confMatrix(X_test, y_test)







