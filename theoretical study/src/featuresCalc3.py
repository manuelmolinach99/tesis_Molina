# -*- coding: utf-8 -*-

import numpy as np
import pickle
from joblib import dump
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append('../lib')
import functions as f

# Load data
file = open('../data/MyoArmband_data_exp1_cor.pickle', 'rb')
ventanas_admitidas = pickle.load(file)
file.close()

# Load data
file = open('../data/MyoArmband_data_exp1_cor.pickle', 'rb')
MyoArm_data = pickle.load(file)
file.close()

# Defino los parametros que voy a usar
fs = MyoArm_data['S1']['info']['fs'] # Frecuency in Hz
nOfSubjects = 2 # N° of Subjects [1,...,10]- cantidad de sujetos con lo que voy a trabajar
nChannels = 16 # N° of Channels- numero de canales con los que quiero trabajar- el maximo es MyoArm_data['S1']['emg'].shape[1]+1
w_ms = 1000 # Window length in ms
s_ms = 50 # Window increment in ms
W_s = int(fs*w_ms/1000) # Window length in samples
I_s = int(fs*s_ms/1000) # Window increment in samples

# Extraigo los datos
data_list = []  # en cada elemento de la lista guardo los datos de un sujeto, luego los concateno
label_list = [] # lo mismo que para la anterior
suject_list = []   # aqui se indica a que sujeto pertenece la senal a lo largo del tiempo

for i in range(1, nOfSubjects+1):
    data_list.append(MyoArm_data['S'+str(i)]['emg'])
    label_list.append(MyoArm_data['S'+str(i)]['label'])
    suject_list.append(MyoArm_data['S'+str(i)]['label']*0 + i)

# Ahora tengo que concatenar los elementos de la lista
data = []
label = []
suject = []
for i in range(len(data_list)):
    data.extend(data_list[i])
    label.extend(label_list[i])
    suject.extend(suject_list[i])
    
data = np.array(data)
label = np.array(label)
suject = np.array(suject)

l = len(ventanas_admitidas)

zcThreshold = 0.05
sscThreshold = 0.05
rms = np.zeros(l)
zc = np.zeros(l)
ssc = np.zeros(l)
wl = np.zeros(l)
mav = np.zeros(l)
ls = np.zeros(l)
mfl = np.zeros(l)
msr = np.zeros(l)
wamp = np.zeros(l)
iav = np.zeros(l)
dasdv = np.zeros(l)
_var = np.zeros(l)
DWT1 = np.zeros(l)
DWT2 = np.zeros(l)
DWT3 = np.zeros(l)

FEATURES = pd.DataFrame()
for j in range(nChannels):
    for i in range(l):
        wl[i] = f.wl(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        zc[i] = f.zc(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]], zcThreshold)
        ssc[i] = f.ssc(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]], sscThreshold)
        rms[i] = f.rms(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        mav[i] = f.mav(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        ls[i] = f.ls(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        mfl[i] = f.mfl(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        msr[i] = f.msr(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        wamp[i] = f.wamp(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        iav[i] = f.iav(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        dasdv[i] = f.dasdv(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
        _var[i] = f._var(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])        
        DWT1[i],DWT2[i],DWT3[i] = f.DWT(data[:,j][ventanas_admitidas[i][0]:ventanas_admitidas[i][1]])
    FEATURES["wl_c"+str(j+1)] = wl/max(abs(wl))
    FEATURES["zc_c"+str(j+1)] = zc/max(abs(zc))  
    FEATURES["ssc_c"+str(j+1)] = ssc/max(abs(ssc))  
    FEATURES["rms_c"+str(j+1)] = rms/max(abs(rms))  
    FEATURES["mav_c"+str(j+1)] = mav/max(abs(mav))
    FEATURES["ls_c"+str(j+1)] = ls/max(abs(ls))
    FEATURES["mfl_c"+str(j+1)] = mfl/max(abs(mfl))  
    FEATURES["msr_c"+str(j+1)] = msr/max(abs(msr))
    FEATURES["wamp_c"+str(j+1)] = wamp/max(abs(min(wamp)), max(wamp))  
    FEATURES["iav_c"+str(j+1)] = iav/max(abs(iav))
    FEATURES["dasdv_c"+str(j+1)] = dasdv/max(abs(dasdv)) 
    FEATURES["var_c"+str(j+1)] = _var/max(abs(_var)) 
    FEATURES["DWT1_c"+str(j+1)] = DWT1/max(abs(DWT1)) 
    FEATURES["DWT2_c"+str(j+1)] = DWT2/max(abs(DWT2)) 
    FEATURES["DWT3_c"+str(j+1)] = DWT3/max(abs(DWT3))

ventanas_admitidas = np.array(ventanas_admitidas)
# Data exportation
labels = {'id_ventana':ventanas_admitidas[:, 2], 'id_gesto':ventanas_admitidas[:, 3], 'subjet': ventanas_admitidas[:, 4]}
labels = pd.DataFrame(labels)
dump(FEATURES, "../data/FEATURES.joblib")
dump(labels, "../data/label.joblib")

plt.show()