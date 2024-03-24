# -*- coding: utf-8 -*-

import scipy.io
import pickle
import numpy as np

# Numero de sujetos
nSubjects = 10

# Lista [S1, S2, ..., S10]
files = ['S' + str(x) for x in range(1,nSubjects+1)] 
data = dict()
data_origin = dict()

# Elijo levantar solo los datos del experimento 1
for file in files:
    E1 = scipy.io.loadmat("../data/" + file + '/' + file + '_E1_A1.mat') # Exercise A
    
    # data es un diccionario que va a tener esta estructura
    data[file] = {'emg':np.concatenate([E1["emg"]]),
                  'label':np.concatenate([E1["stimulus"]]), 'info':{'age':E1['age'], 'gender':E1['gender'], 
                          'n_subjects':10, 'weight':E1['weight'], 
                          'height':E1['height'], 'fs':E1["frequency"][0][0]}, 
                  'description':"Contiene datos de EMG de 10 sujetos, primer experimento por cada sujeto (E1) (A de la figura), en este experimento el sujeto realiza 12 gestos diferentes."}
    
# Guardo el archivo con los datos originales de todos los experimentos
with open("../data/MyoArmband_data_exp1.pickle", "wb") as f:
    pickle.dump(data, f)