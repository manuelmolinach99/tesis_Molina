# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:04:47 2024

@author: Manu
"""

import scipy.io
import matplotlib.pyplot as plt
import pickle

# ruta a los archivos que se quieren levantar
ruta = 'F:/tesis/DB5/s1/'

# nombre del archivo que se quiere levantar
nombre = 'S1_E1_A1.mat'

# Carga el archivo .mat
mat_file1 = scipy.io.loadmat(ruta + nombre)

emg = mat_file1['emg']

archivo_frames = "datos_emg.pkl"

# Escribir la lista en un archivo utilizando pickle
with open(archivo_frames, 'wb') as archivo:
    pickle.dump(emg, archivo)
