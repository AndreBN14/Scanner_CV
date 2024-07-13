import cv2
import numpy as np
import os
import time
import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Variables globales
points = []
captured_images = []
capture_interval = 2  # Intervalo de captura en segundos
total_captures = 20  # Número total de capturas
output_folder = "captured_images"  # Carpeta donde se guardarán las imágenes
background_image = None
angle_x = 0
angle_y = 0
mouse_last_x = 0
mouse_last_y = 0
mouse_down = False

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Función para capturar una imagen de referencia (fondo)
def capture_background_image():
    global background_image
    cap = cv2.VideoCapture(0)  # Abre la cámara (usar el índice apropiado)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer la imagen de la cámara.")
    else:
        background_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Imagen de referencia capturada.")
    cap.release()

# Función para procesar la imagen y extraer los puntos del láser
def process_image(image, angle_deg):
    global background_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resta de fondo
    diff = cv2.absdiff(gray, background_image)

    # Aplicar umbralización
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Filtrar el láser basado en el color (asumiendo láser rojo)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2

    # Combinar máscaras
    combined_mask = cv2.bitwise_and(thresh, mask)

    # Detectar bordes en la máscara combinada
    edges = cv2.Canny(combined_mask, 50, 150)

    # Obtener coordenadas de los puntos
    points = np.column_stack(np.where(edges > 0))

    # Convertir ángulo a radianes
    angle_rad = np.deg2rad(angle_deg)

    # Crear una matriz de rotación alrededor del eje Y
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

    # Convertir puntos y aplicar la rotación
    points = np.array([[point[1] / 100.0, -point[0] / 100.0, 0.0] for point in points])
    points = np.dot(points, rotation_matrix.T)

    return points

# Función de inicialización de OpenGL
def init_gl():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)
    glPointSize(1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

# Función de dibujo de OpenGL
def draw():
    global angle_x, angle_y
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0)

    # Rotación del modelo
    glTranslatef(0.0, 0.0, -1.5)  # Centrar el modelo
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)

    glBegin(GL_POINTS)
    glColor3f(0.0, 1.0, 0.0)  # Color verde
    for point in points:
        glVertex3f(point[0], point[1], point[2])
    glEnd()

    glutSwapBuffers()

# Función para capturar y guardar imágenes a intervalos regulares
def capture_images():
    global captured_images
    cap = cv2.VideoCapture(0)  # Abre la cámara (usar el índice apropiado)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    for i in range(total_captures):
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer la imagen de la cámara.")
            break

        # Guardar la imagen capturada
        image_path = os.path.join(output_folder, f'image_{i + 1}.png')
        cv2.imwrite(image_path, frame)
        captured_images.append(frame)
        print(f"Imagen {i + 1} capturada y guardada en {image_path}.")
        time.sleep(capture_interval)

    cap.release()

# Función para actualizar la visualización
def update_display():
    global points
    points = []
    angle_step = 360 / total_captures  # Incremento del ángulo en grados
    current_angle = 0
    for image in captured_images:
        new_points = process_image(image, current_angle)
        points.extend(new_points)
        current_angle += angle_step
        print(f"Procesados {len(new_points)} puntos de una imagen.")
    glutPostRedisplay()

# Funciones para manejo del mouse
def mouse(button, state, x, y):
    global mouse_down, mouse_last_x, mouse_last_y
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        mouse_down = True
        mouse_last_x = x
        mouse_last_y = y
    elif button == GLUT_LEFT_BUTTON and state == GLUT_UP:
        mouse_down = False

def motion(x, y):
    global angle_x, angle_y, mouse_last_x, mouse_last_y, mouse_down
    if mouse_down:
        dx = x - mouse_last_x
        dy = y - mouse_last_y
        angle_x += dy * 0.5  # Incrementar velocidad de rotación
        angle_y += dx * 0.5  # Incrementar velocidad de rotación
        mouse_last_x = x
        mouse_last_y = y
        glutPostRedisplay()

# Función para manejar las teclas
def key_pressed(*args):
    if args[0] == b'\x1b':  # Escape key
        sys.exit()

# Función principal
def main():
    global points, captured_images

    # Capturar imagen de referencia
    capture_background_image()

    # Capturar imágenes automáticamente
    capture_images()

    if not captured_images:
        print("No se capturaron imágenes.")
        return

    # Inicializar OpenGL
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow('3D Scanner')
    init_gl()
    glutDisplayFunc(draw)
    glutIdleFunc(update_display)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutKeyboardFunc(key_pressed)
    glutMainLoop()

if __name__ == "__main__":
    main()
