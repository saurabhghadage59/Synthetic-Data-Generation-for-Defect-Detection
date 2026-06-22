import os, cv2, shutil
from pathlib import Path

DATA_DIR = Path('data/mvtec/leather')
OUTPUT_DIR = Path('data/lora_training')

defect_types = ['color', 'cut', 'fold', 'glue', 'poke']

for defect in defect_types:
    out_folder = OUTPUT_DIR / f'10_leather_{defect}_defect'
    out_folder.mkdir(parents=True, exist_ok=True)
    
    src = DATA_DIR / 'test' / defect
    count = 0
    for img_path in src.glob('*.png'):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(str(out_folder / img_path.name), img)
        count += 1
    
    print(f'{defect}: {count} images prepared')

print('Done!')