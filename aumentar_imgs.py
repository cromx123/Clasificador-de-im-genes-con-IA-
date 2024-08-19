from PIL import Image, ImageEnhance, ImageOps
import os
import random

# Ruta de la carpeta de imágenes
carpeta_imagenes = "./data/Sanos"
nombre = "Sanos"
# Listar las imágenes en la carpeta
imagenes = os.listdir(carpeta_imagenes)

# Definir la cantidad de imágenesa crear
cantidad_imagenes_deseada = 500
contador_imagenes_creadas = 0

while contador_imagenes_creadas < cantidad_imagenes_deseada:
    # Seleccionar una imagen aleatoriamente
    img = random.choice(imagenes)
    ruta_img = os.path.join(carpeta_imagenes, img)
    imagen = Image.open(ruta_img)
    
    # Zoom
    zoom_factor = random.uniform(0.8, 1.8)
    width, height = imagen.size
    imagen = imagen.crop(((width - width * zoom_factor) // 2,
                          (height - height * zoom_factor) // 2,
                          (width + width * zoom_factor) // 2,
                          (height + height * zoom_factor) // 2))
    imagen = imagen.resize((width, height), Image.LANCZOS)

    # Flip vertical
    if random.random() < 0.7:
        imagen = ImageOps.flip(imagen)

    # Ajuste de brillo
    enhancer = ImageEnhance.Brightness(imagen)
    imagen = enhancer.enhance(random.uniform(0.7, 1.2))
    
    # Aplicar transformaciones condicionales
    if random.random() < 0.6:
        imagen = ImageOps.mirror(imagen)

    if random.random() < 0.6:
        enhancer = ImageEnhance.Brightness(imagen)
        imagen = enhancer.enhance(random.uniform(0.7, 1.3))

    # Ajuste de contraste
    contrast_enhancer = ImageEnhance.Contrast(imagen)
    imagen = contrast_enhancer.enhance(random.uniform(0.7, 1.3))

    # Ajuste de saturación
    color_enhancer = ImageEnhance.Color(imagen)
    imagen = color_enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Guardar la imagen
    nuevo_nombre = f"{nombre}_m_{contador_imagenes_creadas + 1}.png"
    imagen.save(os.path.join(carpeta_imagenes, nuevo_nombre))

    contador_imagenes_creadas += 1

print(f"Se han creado {contador_imagenes_creadas} imágenes nuevas.")