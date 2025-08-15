
import argparse
from typing import Tuple, List
import numpy as np
import cv2

def analyze_image(image_path: str, min_area: int = 300, color_tol: int = 45) -> Tuple[int, int, List[tuple]]:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Маска "не белых" пикселей
    dist_from_white = np.linalg.norm(img.astype(np.int16) - 255, axis=2)
    mask = (dist_from_white > 25).astype(np.uint8)

    # Очистка мелкого шума
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Компоненты связности
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_ids = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    # Средние цвета по каждой фигуре
    colors = []
    for cid in component_ids:
        comp_mask = (labels == cid)
        mean_color = img[comp_mask].mean(axis=0)
        colors.append(tuple(int(round(c)) for c in mean_color))

    # Кластеризация цветов по порогу (боремся с антиалиасингом на границах)
    unique_colors: List[tuple] = []
    for c in colors:
        if not unique_colors:
            unique_colors.append(c)
            continue
        dists = [np.linalg.norm(np.array(c) - np.array(u)) for u in unique_colors]
        if min(dists) > color_tol:
            unique_colors.append(c)

    return len(component_ids), len(unique_colors), unique_colors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Посчитать количество фигур и уникальных цветов на картинке (на белом фоне).")
    parser.add_argument("image_path", help="Путь к изображению (png/jpg и т.п.)")
    parser.add_argument("--min-area", type=int, default=300, help="Минимальная площадь компоненты (отсечь шум)")
    parser.add_argument("--color-tol", type=int, default=45, help="Порог схожести цветов (0-~60) для объединения близких оттенков")
    args = parser.parse_args()

    shapes, colors_n, colors = analyze_image(args.image_path, args.min_area, args.color_tol)
    print(f"Фигур: {shapes}")
    print(f"Цветов: {colors_n}")
    print("Цвета (RGB):", colors)
