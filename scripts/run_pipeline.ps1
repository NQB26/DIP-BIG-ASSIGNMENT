# Chạy pipeline tích hợp (TV3) trên video mẫu. Chạy từ thư mục gốc của dự án.
#   .\scripts\run_pipeline.ps1 -Rules .\configs\rules.yaml -Out .\runs\demo_all
param(
    [string]$Source = ".\data\zone_test.mp4",
    [string]$Person = ".\yolo11n.pt",
    [string]$Coat   = ".\runs\detect\runs\detect\labcoat_v3_baseline\weights\best.pt",
    [string]$Zones  = ".\configs\zones.yaml",
    [string]$Rules  = ".\configs\rules.yaml",
    [string]$Out    = ".\runs\demo_all",
    [int]$MaxFrames = 0,
    [switch]$Show
)
$argsList = @(
    ".\src\pipeline\runner.py",
    "--source", $Source, "--weights", $Person, "--lab-coat-weights", $Coat,
    "--zones", $Zones, "--rules", $Rules, "--out", $Out,
    "--imgsz", "512", "--device", "cpu", "--coco-conf", "0.25", "--coat-conf", "0.25",
    "--max-frames", $MaxFrames
)
if ($Show) { $argsList += "--show" }
python @argsList
