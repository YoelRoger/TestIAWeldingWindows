import os
from ultralytics import YOLO

# Ruta al modelo entrenado (ajusta si el nombre o la ubicación es diferente)
model_path = r'C:\Users\yoelf\Downloads\Welding_IA\runs\detect\welding_v1_aug14\weights\best.pt'

# Ruta al directorio de imágenes de prueba
test_images_dir = r'C:\Users\yoelf\Downloads\Welding_IA\tassaroli'

# Directorio donde se guardarán las imágenes con detecciones
output_dir = r'C:\Users\yoelf\Downloads\Welding_IA\detections'
os.makedirs(output_dir, exist_ok=True)

# Cargar el modelo YOLO
print(f"Cargando modelo desde: {model_path}")
model = YOLO(model_path)


# Función para realizar detecciones
def test_model(model, images_dir, output_dir):
    # Listar todas las imágenes en el directorio de prueba
    images = [img for img in os.listdir(images_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]

    for image_name in images:
        image_path = os.path.join(images_dir, image_name)

        # Realizar la detección
        print(f"Procesando: {image_path}")
        results = model(image_path)

        # Guardar la imagen con los resultados
        for result in results:
            output_path = os.path.join(output_dir, image_name)
            result.plot(save=True, filename=output_path)
            print(f"Detección guardada en: {output_path}")


# Ejecutar detección
test_model(model, test_images_dir, output_dir)
print("Detecciones completadas.")
