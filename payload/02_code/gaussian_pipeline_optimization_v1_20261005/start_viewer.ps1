param(
    [Parameter(Mandatory=$true)][string]$Pair,
    [int]$Port = 8196,
    [ValidatePattern('^[A-Za-z0-9_]+$')][string]$InitialCase = 'full'
)
$ErrorActionPreference = 'Stop'
$viewerRoot = (Resolve-Path -LiteralPath (Join-Path $Pair 'viewer')).Path
if (-not (Test-Path -LiteralPath (Join-Path $viewerRoot 'EXPORT_RECEIPT.json'))) {
    throw 'The directory is missing its model export receipt.'
}
if (-not (Test-Path -LiteralPath (Join-Path $viewerRoot "cases/$InitialCase/index.html"))) {
    throw 'The declared initial viewer case is missing.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is occupied. Verify its owner before reusing it."
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '05_envs/sfm_py311_v1/Scripts/python.exe'
$serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u', '-m', 'http.server', $Port, '--bind', '127.0.0.1') -WorkingDirectory $viewerRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $viewerRoot 'server.stdout.log') -RedirectStandardError (Join-Path $viewerRoot 'server.stderr.log')
$receipt = @{ status='started_pending_http_check'; pid=$serverProcess.Id; root=$viewerRoot; port=$Port; url="http://127.0.0.1:$Port/cases/$InitialCase/index.html" }
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $viewerRoot 'SERVER_STATE.json') -Encoding utf8
$receipt | ConvertTo-Json
