# Mở dashboard cảnh báo tại http://127.0.0.1:8000 (không cần cài thêm thư viện).
param([string]$RunsDir = ".\runs", [int]$Port = 8000)
Start-Process "http://127.0.0.1:$Port"
python .\src\dashboard\server.py --runs-dir $RunsDir --port $Port
