param(
    [Parameter(Mandatory=$true)][string]$ViewerRoot,
    [int]$Port = 8197
)
$ErrorActionPreference = 'Stop'
$resolvedViewer = (Resolve-Path -LiteralPath $ViewerRoot).Path
$contractFile = Join-Path $resolvedViewer 'CONTRACT.json'
$contract = Get-Content -LiteralPath $contractFile -Raw | ConvertFrom-Json
if (-not $contract.scene_asset_identity_equal) {
    throw 'The paired viewer assets have not passed their identity check.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is occupied. Verify its owner before reusing it."
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '05_envs/sfm_py311_v1/Scripts/python.exe'
$serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u', '-m', 'http.server', $Port, '--bind', '127.0.0.1') -WorkingDirectory $resolvedViewer -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $resolvedViewer 'server.stdout.log') -RedirectStandardError (Join-Path $resolvedViewer 'server.stderr.log')
$receipt = @{ status='started_pending_http_and_browser_check'; pid=$serverProcess.Id; root=$resolvedViewer; port=$Port; url="http://127.0.0.1:$Port/cases/demand/index.html" }
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedViewer 'SERVER_STATE.json') -Encoding utf8
$receipt | ConvertTo-Json
