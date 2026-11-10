param(
    [Parameter(Mandatory=$true)][string]$ViewerRoot,
    [int]$Port = 8203
)
$ErrorActionPreference = 'Stop'
$resolvedViewer = (Resolve-Path -LiteralPath $ViewerRoot).Path
$contract = Get-Content -LiteralPath (Join-Path $resolvedViewer 'CONTRACT.json') -Raw | ConvertFrom-Json
if ($contract.mechanism -ne 'FINAL_NEW_ROUTE_COMBINATION') { throw 'Missing new-route display contract.' }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw 'Port is already occupied.' }
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '05_envs/sfm_py311_v1/Scripts/python.exe'
$serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u', '-m', 'http.server', $Port, '--bind', '127.0.0.1') -WorkingDirectory $resolvedViewer -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $resolvedViewer 'server.stdout.log') -RedirectStandardError (Join-Path $resolvedViewer 'server.stderr.log')
$receipt = @{ status='started_pending_http_and_browser_validation'; pid=$serverProcess.Id; root=$resolvedViewer; port=$Port; url="http://127.0.0.1:$Port/combined/cases/packed/index.html?mode=box" }
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedViewer 'SERVER_STATE.json') -Encoding utf8
$receipt | ConvertTo-Json
