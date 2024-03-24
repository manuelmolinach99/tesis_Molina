# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:20:20 2024

@author: Ing. Biologica
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


file = open('MyoArmband_data_exp1.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()

# Numero de muestras de la correccion
correc = 200

new_emg = []    # Creo una lista para guardar los datos de emg corregidos
new_label = []  # Creo una lista para guardar las etiquetas corregidas

for j in range(len(MyoArm_data)):
    EMG = MyoArm_data['S'+str(j+1)]['emg']  # tengo 16 se;ales de emg aca
    etiqueta = MyoArm_data['S'+str(j+1)]['label']
    etiqueta = etiqueta[:, 0]
    indices_ventanas = np.where(np.diff(etiqueta)!=0)[0]
    
    # Ahora tengo que chequear la senal no comience o termine en la mitad de un gesto, en caso positivo lo tiro    
    # Chequeo el comienzo
    if etiqueta[0] != 0:
        etiqueta = etiqueta[(indices_ventanas[0]+1):]    # Recorto la se;al hasta el indice donde termina la ventana que estoy tirando
        EMG = EMG[(indices_ventanas[0]+1):, :]
        indices_ventanas = np.where(np.diff(etiqueta)!=0)[0]
    
    # Chequeo el fin
    if etiqueta[-1] != 0:
        etiqueta = etiqueta[:(indices_ventanas[-1])]    # Recorto la se;al hasta el indice donde termina la ventana que estoy tirando
        EMG = EMG[:(indices_ventanas[-1]), :]
        indices_ventanas = indices_ventanas[:-1]
        
    # Procedo a hacer la correccion
    for i in range(int(len(indices_ventanas)/2)):     # El largo/2 da la cantidad de ventanas
        etiqueta[indices_ventanas[2*i]: (indices_ventanas[2*i]+correc)] = 0            # indices_ventanas[2*i] se para en cada lugar que arranca una ventana (indices pares)    
        etiqueta[((indices_ventanas[2*i+1])-correc+1):(indices_ventanas[2*i+1])+1] = 0     # indices_ventanas[2*i+1] se para en los finales de cada ventana.
        
    # Procedo a hacer la normalizacion
    # Normalizacion con distribucion normal. Lo voy a hacer por columna
    for i in range(EMG.shape[1]):
        EMG[:, i] = (EMG[:, i] - np.mean(EMG[:, i]))/np.std(EMG[:, i])
    
    # Modifico el archivo de datos
    MyoArm_data['S'+str(j+1)]['emg'] = EMG
    MyoArm_data['S'+str(j+1)]['label'] = etiqueta
    
        
# EMG = MyoArm_data['S1']['emg']
# etiqueta = MyoArm_data['S1']['label']

# # Visualizacion de datos crudos s1
# plt.figure()
# plt.plot(EMG[:, 0])
# plt.plot(etiqueta)

with open("../data/MyoArmband_data_exp1_cor.pickle", "wb") as f:
    pickle.dump(MyoArm_data, f)
        
        
    
    