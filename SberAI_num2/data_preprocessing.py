import os
import cv2
from pathlib import Path
train_dir = Path("dataset/train")
image_paths = list(train_dir.rglob("*.png"))
for img_path in image_paths:
    path_str = str(img_path)
    image = cv2.imread(path_str)
    if image is None:
        os.remove(path_str)
        continue
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Всего найдено PNG файлов: {len(image_paths)}")