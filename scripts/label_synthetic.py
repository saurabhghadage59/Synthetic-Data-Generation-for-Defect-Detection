from pathlib import Path

SYN_IMG_DIR = Path('data/synthetic/images')
SYN_LBL_DIR = Path('data/synthetic/labels')

CLASSES = {'color': 0, 'cut': 1, 'fold': 2, 'glue': 3, 'poke': 4}

for defect, class_id in CLASSES.items():
    lbl_dir = SYN_LBL_DIR / defect
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img_dir = SYN_IMG_DIR / defect
    count = 0
    for img_path in img_dir.glob('*.png'):
        label = f'{class_id} 0.5 0.5 0.5 0.5'
        (lbl_dir / img_path.stem).with_suffix('.txt').write_text(label)
        count += 1
    print(f'{defect}: {count} labels created')

print('Synthetic labels done!')