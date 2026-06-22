import cv2, numpy as np, os
from pathlib import Path

LEATHER_DIR = Path('data/mvtec/leather')
REAL_IMG_DIR = Path('data/real/images')
REAL_LBL_DIR = Path('data/real/labels')
REAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
REAL_LBL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {'color': 0, 'cut': 1, 'fold': 2, 'glue': 3, 'poke': 4}

for defect, class_id in CLASSES.items():
    img_dir = LEATHER_DIR / 'test' / defect
    mask_dir = LEATHER_DIR / 'ground_truth' / defect

    for img_path in img_dir.glob('*.png'):
        img = cv2.imread(str(img_path))
        img_resized = cv2.resize(img, (640, 640))
        mask_path = mask_dir / img_path.name.replace('.png', '_mask.png')

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask, (640, 640))
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
                cx = (x + bw/2) / 640
                cy = (y + bh/2) / 640
                nw = bw / 640
                nh = bh / 640
                label = f'{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}'
            else:
                label = f'{class_id} 0.5 0.5 0.4 0.4'
        else:
            label = f'{class_id} 0.5 0.5 0.4 0.4'

        stem = f'{defect}_{img_path.stem}'
        cv2.imwrite(str(REAL_IMG_DIR / f'{stem}.png'), img_resized)
        (REAL_LBL_DIR / f'{stem}.txt').write_text(label)

print(f'Real dataset: {len(list(REAL_IMG_DIR.glob("*.png")))} images with labels')