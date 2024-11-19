import os
import shutil
from ultralytics import YOLO
import wandb
import torchvision
# Iniciar sesión en WandB con la clave proporcionada
wandb.login(key='eb4c4a1fa7eec1ffbabc36420ba1166f797d4ac5')

def copy_dataset_to_yolo(src_dir, dest_dir):
    """
    Copia un dataset desde src_dir a dest_dir, asegurándose de que todos los archivos existen.
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"El directorio fuente no existe: {src_dir}")

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)  # Borra el destino si ya existe
    os.makedirs(dest_dir, exist_ok=True)

    # Copiar archivos de forma segura
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_path, src_dir)
            dest_path = os.path.join(dest_dir, relative_path)

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Imprimir rutas generadas
            print(f"Intentando copiar: {src_path} -> {dest_path}")
            try:
                shutil.copy2(src_path, dest_path)
            except FileNotFoundError:
                print(f"Advertencia: El archivo no se encontró y no se copió: {src_path}")

# Rutas absolutas
base_dir = os.path.abspath("C:/Users/yoelf/Downloads/Welding_IA")  # Ruta base del proyecto
v1_dir = os.path.join(base_dir, "dataset welding", "The Welding Defect Dataset", "The Welding Defect Dataset")
v2_dir = os.path.join(base_dir, "dataset welding", "The Welding Defect Dataset - v2", "The Welding Defect Dataset - v2")

yolo_v1_dir = os.path.join(base_dir, "yolov8", "data", "welding_v1")
yolo_v2_dir = os.path.join(base_dir, "yolov8", "data", "welding_v2")

# Copiar datasets
copy_dataset_to_yolo(v1_dir, yolo_v1_dir)
copy_dataset_to_yolo(v2_dir, yolo_v2_dir)

# Actualizar `data.yaml` con rutas absolutas
def update_data_yaml(data_yaml_path, base_dir):
    """
    Actualiza las rutas en el archivo data.yaml para que apunten al dataset correcto.
    """
    train_path = os.path.join(base_dir, "train", "images")
    val_path = os.path.join(base_dir, "valid", "images")
    test_path = os.path.join(base_dir, "test", "images")

    with open(data_yaml_path, 'w') as file:
        file.write(
            f"train: {train_path}\n"
            f"val: {val_path}\n"
            f"test: {test_path}\n\n"
            f"nc: 3\n"
            f"names: ['Bad Weld', 'Good Weld', 'Defect']\n"
        )
    print(f"Archivo data.yaml actualizado en: {data_yaml_path}")

# Rutas para los YAML de cada versión
yolo_v1_yaml_path = os.path.join(yolo_v1_dir, "data.yaml")
yolo_v2_yaml_path = os.path.join(yolo_v2_dir, "data.yaml")

update_data_yaml(yolo_v1_yaml_path, yolo_v1_dir)
update_data_yaml(yolo_v2_yaml_path, yolo_v2_dir)

# Verificar contenido de data.yaml
def print_data_yaml(data_yaml_path):
    with open(data_yaml_path, 'r') as file:
        print(file.read())

print("Contenido de data.yaml para la versión 1:")
print_data_yaml(yolo_v1_yaml_path)

print("Contenido de data.yaml para la versión 2:")
print_data_yaml(yolo_v2_yaml_path)

# Entrenar modelos YOLOv8
model_v1 = YOLO('yolov8m.pt')
model_v2 = YOLO('yolov8m.pt')

model_v1.train(
    data=yolo_v1_yaml_path,
    epochs=100,
    imgsz=640,
    batch=6,
    device=0,
    workers=0,  # Esto evita el uso de múltiples procesos
    name='welding_v1_aug',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=0.05,
    cls=0.5,
    iou=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    mixup=0.5
)

"""model_v2.train(
    data=yolo_v2_yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    name='welding_v2_aug',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=0.05,
    cls=0.5,
    iou=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    mixup=0.5
)
"""