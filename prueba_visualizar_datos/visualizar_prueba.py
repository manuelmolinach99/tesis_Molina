# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:05:54 2024

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
tiempo = np.linspace(0, 10, 1000)
amplitud_ecg = np.sin(2 * np.pi * 1 * tiempo) + 0.5 * np.sin(2 * np.pi * 2.5 * tiempo)
desplazamiento_tiempo = 0

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

    # Actualizar la señal de ECG (simulación de datos en tiempo real)
    desplazamiento_tiempo += 0.01
    amplitud_ecg = np.sin(2 * np.pi * 1 * (tiempo + desplazamiento_tiempo)) + 0.5 * np.sin(2 * np.pi * 2.5 * (tiempo + desplazamiento_tiempo))

    # Limpiar la pantalla
    ventana.fill(NEGRO)

    # Dibujar el eje de tiempo
    pygame.draw.line(ventana, BLANCO, (50, ALTO - 30), (ANCHO - 50, ALTO - 30), 2)  # Eje de tiempo

    # Calcular el rango de tiempo mostrado en la ventana (ahora 5 segundos)
    tiempo_actual = desplazamiento_tiempo + ventana_tiempo / 2  # Tiempo actual centrado en la ventana
    inicio_tiempo = tiempo_actual - ventana_tiempo / 2
    fin_tiempo = tiempo_actual + ventana_tiempo / 2

    # Calcular el nivel de aclarado gradual
    aclarado = np.linspace(1, nivel_minimo_aclarado, len(tiempo))  # De 1 (totalmente negro) al nivel mínimo de aclarado

    # Dibujar la señal de ECG con nivel de aclarado gradual
    for i in range(len(tiempo) - 1):
        color = (CELESTE[0] * (1 - aclarado[i]), CELESTE[1] * (1 - aclarado[i]), CELESTE[2] * (1 - aclarado[i]))  # Modificar el color celeste según el nivel de aclarado
        pygame.draw.line(ventana, color, (int(i * (ANCHO - 100) / len(tiempo)) + 50, int(ALTO / 2 - amplitud_ecg[i] * ALTO / 4)),
                         (int((i + 1) * (ANCHO - 100) / len(tiempo)) + 50, int(ALTO / 2 - amplitud_ecg[i + 1] * ALTO / 4)), grosor_linea)

    # Dibujar los números en el eje de tiempo
    for t in np.linspace(inicio_tiempo, fin_tiempo, 11):
        texto = pygame.font.SysFont(None, 24).render(f"{t:.1f}", True, BLANCO)
        posicion_x = 50 + (ANCHO - 100) * (t - inicio_tiempo) / ventana_tiempo - texto.get_width() / 2
        ventana.blit(texto, (posicion_x, ALTO - 25))

    # Dibujar el eje de amplitud
    for y in np.arange(-1.5, 2, intervalo_amplitud):
        # Calcular la posición vertical de la línea en la ventana
        posicion_y = int(ALTO / 2 - y * ALTO / 4)
        # Dibujar la línea horizontal
        pygame.draw.line(ventana, BLANCO, (40, posicion_y), (50, posicion_y), 2)
        # Dibujar el texto con la amplitud
        texto = pygame.font.SysFont(None, 24).render(f"{y:.1f}", True, BLANCO)
        ventana.blit(texto, (5, posicion_y - texto.get_height() / 2))

    # Actualizar la pantalla
    pygame.display.flip()

    # Esperar un tiempo para mantener la velocidad de la animación
    reloj.tick(60)

# Salir de Pygame
pygame.quit()















