# val.ps1
# Validation script for a trained checkpoint.
# Usage:
# powershell -ExecutionPolicy Bypass -File .\scripts\val.ps1 -Weights C:/.../best.pt

param(
  [Parameter(Mandatory = $true)]
  [string]$Weights,
  [int]$ImgSz = 640,
  [ValidateSet('train','val','test')]
  [string]$Split = 'val'
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path.Replace('\','/')
$dataYaml = "$projectRoot/configs/dataset.yaml"
$env:YOLO_CONFIG_DIR = "$projectRoot/.yolo"
$env:ULTRALYTICS_SETTINGS_DIR = "$projectRoot/.yolo"
$yoloCmd = Get-Command yolo -ErrorAction SilentlyContinue
if (-not $yoloCmd) {
  throw "yolo command not found. Run: pip install ultralytics"
}
$yoloExe = $yoloCmd.Source
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

& $yoloExe detect val `
  model=$Weights `
  data=$dataYaml `
  imgsz=$ImgSz `
  split=$Split `
  project=$projectRoot/runs `
  name=${Split}_$(Get-Date -Format 'yyyyMMdd_HHmm')
