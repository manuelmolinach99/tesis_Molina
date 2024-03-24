# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np


file = open('MyoArmband_data_exp1.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()
correc = 200
EMG = MyoArm_data['S1']['emg']
etiqueta = MyoArm_data['S1']['label']
fs = MyoArm_data['S1']['info']['fs']
t = np.arange(len(EMG))/fs


# Voy a trabajar con el sujeto 1, y con las se;ales 1 y 5 al igual que en emgAnalysis
s1 = EMG[:, 0]
s5 = EMG[:, 4]

# #Visualizacion de datos crudos s1
# plt.figure()
# plt.plot(t, s1)
# plt.plot(t, etiqueta)

# Normalizacion con el maximo y quitando maximo
s1 = (s1 - np.mean(s1))/np.max(s1)
s5 = (s5 - np.mean(s5))/np.max(s5)

# Creacion de una variable que indique las ventanas de tiempo en las que hay gesto
ventanas_gestos = etiqueta>=1
ventanas_gestos_aux = ventanas_gestos.flatten()
ventanas_gestos = np.zeros(len(ventanas_gestos_aux))

# Lo anterior era bool, ahora lo paso a enteros (True=1)
for i in range(0, len(ventanas_gestos)):
    ventanas_gestos[i] = int(ventanas_gestos_aux[i])


# Voy a comenzar con la s1

# Visualizacion preliminar
plt.figure()
plt.plot(s1)
plt.plot(ventanas_gestos)
plt.legend(['Sig 1', 'ventanas de gestos'])

# Ahora voy a aplicar una correccion de 200 muestras a las etiquetas de gestos, lo que equivale a 1s
# Las primeras y ultimas 200 muestras de cada ventana sera llevada a 0

# Hago un dif de ventanas_gestos para tener los indices de principio y fin de cada ventana
# Me quedo con los lugares distintos de 0
indices_ventanas_gestos = np.where(np.diff(ventanas_gestos)!=0)[0]
indices = np.where(np.diff(etiqueta[:, 0])!=0)[0]

# Chequeo que la se;al de etiqueta inicie y finalice en 0 para que no haya gestos por la mitad
# Si algo de esto no se cumple, tiro el segmento de se;al que haga falta

#%%
if ventanas_gestos[0] != 0:
    ventanas_gestos = ventanas_gestos[(indices_ventanas_gestos[0]+1):]    # Recorto la se;al hasta el indice donde termina la ventana que estoy tirando
    s1 = s1[(indices_ventanas_gestos[0]+1):]
    indices_ventanas_gestos = np.where(np.diff(ventanas_gestos)!=0)[0]
    
# # Visualizacion preliminar
# plt.figure()
# plt.plot(s1)
# plt.plot(ventanas_gestos)
# plt.legend(['Sig 1', 'ventanas de gestos'])

if ventanas_gestos[-1] != 0:
    ventanas_gestos = ventanas_gestos[:(indices_ventanas_gestos[-1])]    # Recorto la se;al hasta el indice donde termina la ventana que estoy tirando
    s1 = s1[:(indices_ventanas_gestos[-1])]
    indices_ventanas_gestos = indices_ventanas_gestos[:-1]

# Visualizacion preliminar
plt.figure() 
plt.plot(s1)
plt.plot(ventanas_gestos)
plt.legend(['Sig 1', 'ventanas de gestos'])

# Primero voy a hacerlo directamente con los indices, luego sobre las ventanas a ver si me da lo mismo
# Recortar ventanas
ind_ventanas_gestos_cor = indices_ventanas_gestos.copy()
ind_ventanas_gestos_cor[::2] = ind_ventanas_gestos_cor[::2] + correc
ind_ventanas_gestos_cor[1::2] = ind_ventanas_gestos_cor[1::2] - correc

# Visualizacion preliminar
plt.figure() 
plt.plot(s1)
plt.plot(ventanas_gestos)
plt.legend(['Sig 1', 'ventanas de gestos'])

#%% Recorte de ventanas sobre ventanas_gestos

ventanas_gestos_aux = ventanas_gestos.copy()

for i in range(int(len(indices_ventanas_gestos)/2)):     # El largo/2 da la cantidad de ventanas
    ventanas_gestos_aux[indices_ventanas_gestos[2*i]: (indices_ventanas_gestos[2*i]+correc)+1] = 0            # indices_ventanas_gestos[2*i] se para en cada lugar que arranca una ventana (indices pares)    
    ventanas_gestos_aux[((indices_ventanas_gestos[2*i+1])-correc+1):(indices_ventanas_gestos[2*i+1])+1] = 0     # indices_ventanas_gestos[2*i+1] se para en los finales de cada ventana.

# Los +1 al final de cada indice son para corregir desajustes provocados por el hecho de que no se incluye la muestra que marca el final del intervalo

# Visualizacion preliminar
plt.figure() 
plt.plot(s1)
plt.plot(ventanas_gestos_aux)
plt.plot(ventanas_gestos)
plt.legend(['Sig 1', 'ventanas corregidas', 'sin corregir'])

# Ahora comparemos con la estrategia de usar los indices
indices_ventanas_gestos_aux = np.where(np.diff(ventanas_gestos_aux)!=0)[0]
# Me dio lo mismo.





