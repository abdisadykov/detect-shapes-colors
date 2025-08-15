# detect-shapes-colors

Скрипт для подсчёта фигур и уникальных цветов на изображении с белым фоном.

## Необходимые команды
```bash
pip install opencv-python numpy
python detect_shapes_colors.py image.png

```
Опции (при необходимости):
```bash

--min-area 300      # отсечка шума по площади
--color-tol 45      # порог схожести цветов (увеличь, если оттенков слишком много)
