import shutil, random
from pathlib import Path

random.seed(42)

REAL_IMGS = list(Path('data/real/images').glob('*.png'))
random.shuffle(REAL_IMGS)
split = int(0.8 * len(REAL_IMGS))
real_train, real_val = REAL_IMGS[:split], REAL_IMGS[split:]

def make_dirs(base):
    for s in ['train', 'val']:
        (base / 'images' / s).mkdir(parents=True, exist_ok=True)
        (base / 'labels' / s).mkdir(parents=True, exist_ok=True)

def copy_pair(img, img_dst, lbl_src, lbl_dst):
    shutil.copy(img, img_dst)
    lbl = (lbl_src / img.stem).with_suffix('.txt')
    if lbl.exists(): shutil.copy(lbl, lbl_dst)

# 1. real_only
base = Path('data/real_only')
make_dirs(base)
for img in real_train:
    copy_pair(img, base/'images'/'train', Path('data/real/labels'), base/'labels'/'train')
for img in real_val:
    copy_pair(img, base/'images'/'val', Path('data/real/labels'), base/'labels'/'val')
print(f"real_only — train: {len(real_train)}, val: {len(real_val)}")

# 2. synthetic_only
base = Path('data/synthetic_only')
make_dirs(base)
SYN_IMGS = []
for d in ['color','cut','fold','glue','poke']:
    SYN_IMGS.extend(list(Path(f'data/synthetic/images/{d}').glob('*.png')))
random.shuffle(SYN_IMGS)
syn_split = int(0.8 * len(SYN_IMGS))
syn_train, syn_val = SYN_IMGS[:syn_split], SYN_IMGS[syn_split:]
for img in syn_train:
    copy_pair(img, base/'images'/'train', Path(f'data/synthetic/labels/{img.parent.name}'), base/'labels'/'train')
for img in syn_val:
    copy_pair(img, base/'images'/'val', Path(f'data/synthetic/labels/{img.parent.name}'), base/'labels'/'val')
print(f"synthetic_only — train: {len(syn_train)}, val: {len(syn_val)}")

# 3. mixed
base = Path('data/mixed')
make_dirs(base)
for img in real_train:
    copy_pair(img, base/'images'/'train', Path('data/real/labels'), base/'labels'/'train')
for img in syn_train:
    copy_pair(img, base/'images'/'train', Path(f'data/synthetic/labels/{img.parent.name}'), base/'labels'/'train')
for img in real_val:
    copy_pair(img, base/'images'/'val', Path('data/real/labels'), base/'labels'/'val')
print(f"mixed — train: {len(real_train)+len(syn_train)}, val: {len(real_val)}")

print('\nAll 3 splits built!')