from ultralytics import YOLO
import pandas as pd, os

def main():
    experiments = {
        'real_only':      'runs/detect/results/exp1_real_only-2/weights/best.pt',
        'synthetic_only': 'runs/detect/results/exp2_synthetic_only-2/weights/best.pt',
        'mixed':          'runs/detect/results/exp3_mixed/weights/best.pt',
    }

    os.makedirs('results', exist_ok=True)
    results = {}

    for name, weights in experiments.items():
        model = YOLO(weights)
        metrics = model.val(data='real_only.yaml', imgsz=640, batch=4, workers=0, verbose=False)
        results[name] = {
            'mAP50':     round(metrics.box.map50, 4),
            'mAP50-95':  round(metrics.box.map, 4),
            'Precision': round(metrics.box.mp, 4),
            'Recall':    round(metrics.box.mr, 4),
        }
        print(f"{name}: mAP@0.5 = {results[name]['mAP50']}")

    df = pd.DataFrame(results).T
    print('\n', df)
    df.to_csv('results/comparison.csv')
    print('Saved to results/comparison.csv')

if __name__ == '__main__':
    main()