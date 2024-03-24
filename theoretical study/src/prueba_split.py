# -*- coding: utf-8 -*-

# La entrada de este archivo son las features que calcule en featuresCalc.py
# Tengo los valores en FEATURES.joblib, y las etiquetas en label.joblib
# Quiero poder saber a que sujeto corresponde cada ventana sobre la que se
# calcularon features para poder hacer un correcto split.

from joblib import load
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../lib')
import classes as cl

FEATURES = load("../data/FEATURES.joblib")
label = load("../data/label.joblib")
# data = cl.data(features = FEATURES, labels = label, chNum = 16, featureNum = 15)
