# infer.ps1
# Inference script for image folder visualization.
# Usage:
# powershell -ExecutionPolicy Bypass -File .\scripts\infer.ps1 -Weights C:/.../best.pt -Source C:/.../images/test

param(
  [Parameter(Mandatory = $true)]
  [string]$Weights,
  [Parameter(Mandatory = $true)]
  [string]$Source,
  [int]$ImgSz = 640
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.Replace('\','/')
$env:YOLO_CONFIG_DIR = "$projectRoot/.yolo"
$env:ULTRALYTICS_SETTINGS_DIR = "$projectRoot/.yolo"
$yoloCmd = Get-Command yolo -ErrorAction SilentlyContinue
if (-not $yoloCmd) {
  throw "yolo command not found. Run: pip install ultralytics"
}
$yoloExe = $yoloCmd.Source
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

& $yoloExe detect predict `
  model=$Weights `
  source=$Source `
  conf=0.25 `
  iou=0.7 `
  imgsz=$ImgSz `
  save=True `
  project=$projectRoot/runs `
  name=infer_$(Get-Date -Format 'yyyyMMdd_HHmm')
