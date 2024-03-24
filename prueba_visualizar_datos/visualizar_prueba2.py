# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:26:54 2024

@author: Manu
"""

import pygame
import numpy as np

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
ANCHO = 800
ALTO = 400
ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Electrocardiograma")

# Colores
NEGRO = (0, 0, 0)
CELESTE = (0, 191, 255)
BLANCO = (255, 255, 255)

# Variables para la señal de ECG
tiempo_total = np.linspace(0, 10, 1000)  # Definir el tiempo total de la señal
amplitud_ecg_total = tiempo_total/10  # Generar la señal de ECG completa

# Grosor de la línea de la señal de ECG
grosor_linea = 3

# Duración de la ventana de tiempo (5 segundos)
ventana_tiempo = 5

# Nivel mínimo de aclarado
nivel_minimo_aclarado = 0.2

# Intervalo de amplitud para el eje de amplitud
intervalo_amplitud = 0.5

# Bucle principal
ejecutando = True
reloj = pygame.time.Clock()

while ejecutando:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            ejecutando = False

    # Limpiar la pantalla
    ventana.fill(NEGRO)

    # Dibujar el eje de tiempo
    pygame.draw.line(ventana, BLANCO, (50, ALTO - 30), (ANCHO - 50, ALTO - 30), 2)  # Eje de tiempo

    # Calcular el rango de tiempo mostrado en la ventana
    tiempo_actual = pygame.time.get_ticks() / 1000  # Tiempo actual en segundos
    inicio_tiempo = tiempo_actual - ventana_tiempo / 2
    fin_tiempo = tiempo_actual + ventana_tiempo / 2

    # Seleccionar el segmento de la señal dentro del rango de tiempo
    indices_segmento = np.where((tiempo_total >= inicio_tiempo) & (tiempo_total <= fin_tiempo))[0]
    tiempo = tiempo_total[indices_segmento]
    amplitud_ecg = amplitud_ecg_total[indices_segmento]

    # Dibujar la señal de ECG
    for i in range(len(tiempo) - 1):
        color = (CELESTE[0] * (1 - nivel_minimo_aclarado), CELESTE[1] * (1 - nivel_minimo_aclarado), CELESTE[2] * (1 - nivel_minimo_aclarado))
        pygame.draw.line(ventana, color, (int((tiempo[i] - inicio_tiempo) * (ANCHO - 100) / ventana_tiempo) + 50, int(ALTO / 2 - amplitud_ecg[i] * ALTO / 4)),
                         (int((tiempo[i + 1] - inicio_tiempo) * (ANCHO - 100) / ventana_tiempo) + 50, int(ALTO / 2 - amplitud_ecg[i + 1] * ALTO / 4)), grosor_linea)

    # Dibujar los números en el eje de tiempo
    for t in np.linspace(inicio_tiempo, fin_tiempo, 11):
        texto = pygame.font.SysFont(None, 24).render(f"{t:.1f}", True, BLANCO)
        posicion_x = 50 + (ANCHO - 100) * (t - inicio_tiempo) / ventana_tiempo - texto.get_width() / 2
        ventana.blit(texto, (posicion_x, ALTO - 25))

    # Dibujar el eje de amplitud
    for y in np.arange(-10, 20, 1):
        posicion_y = int(ALTO / 2 - y * ALTO / 4)
        pygame.draw.line(ventana, BLANCO, (40, posicion_y), (50, posicion_y), 2)
        texto = pygame.font.SysFont(None, 24).render(f"{y:.1f}", True, BLANCO)
        ventana.blit(texto, (5, posicion_y - texto.get_height() / 2))

    # Actualizar la pantalla
    pygame.display.flip()

    # Esperar un tiempo para mantener la velocidad de la animación
    reloj.tick(60)

# Salir de Pygame
pygame.quit()
