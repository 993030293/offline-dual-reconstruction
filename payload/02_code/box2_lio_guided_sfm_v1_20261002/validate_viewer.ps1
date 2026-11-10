param([Parameter(Mandatory=$true)][string]$Run, [int]$Attempt = 1)
$ErrorActionPreference = 'Stop'
$artifactRoot = Join-Path $Run 'output/playwright'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$validationSource = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'validate_viewer.js') -Raw -Encoding UTF8
$validationSource = $validationSource.Replace('__ARTIFACT_ROOT__', $artifactRoot.Replace('\', '/'))
$rawPath = Join-Path $artifactRoot "browser_validation_attempt$Attempt.raw.txt"
if (Test-Path -LiteralPath $rawPath) { throw "Validation output already exists: $rawPath" }
$runtimeScript = Join-Path $artifactRoot "validate_viewer_runtime_attempt$Attempt.js"
$validationSource | Set-Content -LiteralPath $runtimeScript -Encoding UTF8
$cliOutput = & npx.cmd --yes --package '@playwright/cli' playwright-cli '-s=box2-lio-20261002' --raw run-code --filename $runtimeScript
$exitCode = $LASTEXITCODE
$cliOutput | Set-Content -LiteralPath $rawPath -Encoding UTF8
$cliOutput
if ($exitCode -ne 0) { throw "Playwright CLI validation exited with $exitCode" }
