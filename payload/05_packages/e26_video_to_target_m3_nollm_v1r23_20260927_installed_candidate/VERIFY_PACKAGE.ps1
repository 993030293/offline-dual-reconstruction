param(
    [string]$GenericPython = 'python',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
& $GenericPython (Join-Path $PSScriptRoot 'code\verify_package.py') --root $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Package structure verification failed' }

foreach ($script in Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1' -File) {
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $script.FullName,
        [ref]$tokens,
        [ref]$errors
    )
    if ($errors.Count -ne 0) {
        throw "PowerShell syntax error in $($script.Name): $($errors[0].Message)"
    }
}

& $GenericPython (Join-Path $PSScriptRoot 'scripts\package_manifest.py') --root $PSScriptRoot --verify
if ($LASTEXITCODE -ne 0) { throw 'Package manifest verification failed' }
if (-not $SkipTests) {
    $previousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
    $env:PYTHONDONTWRITEBYTECODE = '1'
    try {
        & $GenericPython -B -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -v
        if ($LASTEXITCODE -ne 0) { throw 'Focused package regression tests failed' }
    } finally {
        $env:PYTHONDONTWRITEBYTECODE = $previousNoBytecode
    }
}
Write-Host 'pass'
