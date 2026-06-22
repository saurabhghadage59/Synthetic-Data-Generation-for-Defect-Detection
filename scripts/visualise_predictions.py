from ultralytics import YOLO
import cv2, matplotlib.pyplot as plt
from pathlib import Path
import random

model = YOLO('runs/detect/results/exp3_mixed/weights/best.pt')

# Pick 5 random real test images
test_images = list(Path('data/real_only/images/val').glob('*.png'))
samples = random.sample(test_images, min(5, len(test_images)))

fig, axes = plt.subplots(1, len(samples), figsize=(20, 4))
for ax, img_path in zip(axes, samples):
    results = model.predict(str(img_path), conf=0.25, verbose=False)
    plotted = results[0].plot()  # Returns BGR image with boxes drawn
    ax.imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
    ax.set_title(img_path.stem[:15], fontsize=9)
    ax.axis('off')

plt.suptitle('Mixed Model — Predictions on Real Test Images', fontsize=14)
plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150)
plt.show()
