# LightSign-YOLO

Minimal reproducible code package for traffic sign detection using YOLOv8n.

## Files Included

```text
.
├─ README.md
├─ configs/
│  └─ dataset.yaml
└─ scripts/
   ├─ convert_gtsdb_to_yolo.py
   ├─ train.ps1
   ├─ val.ps1
   └─ infer.ps1
```

## Required Runtime Directories (create when running)

```text
data/gtsdb/images/{train,val,test}
data/gtsdb/labels/{train,val,test}
runs/
```

## 1) Convert Raw GTSDB to YOLO Format

```powershell
python .\scripts\convert_gtsdb_to_yolo.py `
  --src-images C:/path/to/gtsdb/images `
  --gt-file C:/path/to/gtsdb/gt.txt `
  --output-root C:/path/to/repo/data/gtsdb `
  --train-ratio 0.7 `
  --val-ratio 0.2 `
  --test-ratio 0.1 `
  --seed 42
```

Then update `configs/dataset.yaml` fields `path`, `nc`, and `names`.

## 2) Install Dependencies

```powershell
pip install pillow ultralytics
```

## 3) Train

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1
```

## 4) Evaluate

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\val.ps1 -Weights C:/path/to/repo/runs/<train_run>/weights/best.pt -Split test
```

## 5) Inference Visualization

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\infer.ps1 -Weights C:/path/to/repo/runs/<train_run>/weights/best.pt -Source C:/path/to/repo/data/gtsdb/images/test
```
