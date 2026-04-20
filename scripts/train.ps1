# train.ps1
# Baseline training script for LightSign-YOLO.
# Usage:
# powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1
# powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Epochs 20 -ImgSz 640 -Batch 8 -Device cpu

param(
  [int]$Epochs = 20,
  [int]$ImgSz = 640,
  [int]$Batch = 8,
  [string]$Device = "cpu",
  [string]$Model = "yolov8n.pt"
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.Replace('\','/')
$dataYaml = "$projectRoot/configs/dataset.yaml"
$runName = "baseline_yolov8n_$(Get-Date -Format 'yyyyMMdd_HHmm')"
$env:YOLO_CONFIG_DIR = "$projectRoot/.yolo"
$env:ULTRALYTICS_SETTINGS_DIR = "$projectRoot/.yolo"
$yoloCmd = Get-Command yolo -ErrorAction SilentlyContinue
if (-not $yoloCmd) {
  throw "yolo command not found. Run: pip install ultralytics"
}
$yoloExe = $yoloCmd.Source
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

& $yoloExe detect train `
  model=$Model `
  data=$dataYaml `
  imgsz=$ImgSz `
  epochs=$Epochs `
  batch=$Batch `
  device=$Device `
  project=$projectRoot/runs `
  name=$runName `
  pretrained=True `
  seed=42
