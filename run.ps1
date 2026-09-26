<#
.SYNOPSIS
    Windows task runner for the wc-claim-cost project (same targets as the Makefile).

.EXAMPLE
    .\run.ps1 setup
    .\run.ps1 test
    .\run.ps1 lint
    .\run.ps1 format
    .\run.ps1 pipeline -Stage validate
    .\run.ps1 app
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('setup', 'test', 'lint', 'format', 'pipeline', 'app')]
    [string]$Command,

    # Stage names are validated by src/pipeline.py so the list lives in one place.
    [string]$Stage = 'all'
)

$ErrorActionPreference = 'Stop'
$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$CodeDirs = @('src', 'tests', 'app')

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Arguments -join ' ')" }
}

function Assert-Venv {
    if (-not (Test-Path $VenvPython)) { throw "Virtual environment not found. Run '.\run.ps1 setup' first." }
}

Push-Location $PSScriptRoot
try {
    switch ($Command) {
        'setup' {
            if (-not (Test-Path $VenvPython)) { Invoke-Checked 'python' @('-m', 'venv', '.venv') }
            Invoke-Checked $VenvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
            Invoke-Checked $VenvPython @('-m', 'pip', 'install', '-r', 'requirements.txt')
        }
        'test' { Assert-Venv; Invoke-Checked $VenvPython @('-m', 'pytest') }
        'lint' {
            Assert-Venv
            Invoke-Checked $VenvPython (@('-m', 'ruff', 'check') + $CodeDirs)
            Invoke-Checked $VenvPython (@('-m', 'black', '--check') + $CodeDirs)
        }
        'format' {
            Assert-Venv
            Invoke-Checked $VenvPython (@('-m', 'ruff', 'check', '--fix') + $CodeDirs)
            Invoke-Checked $VenvPython (@('-m', 'black') + $CodeDirs)
        }
        'pipeline' { Assert-Venv; Invoke-Checked $VenvPython @('-m', 'src.pipeline', '--stage', $Stage) }
        'app' { Assert-Venv; Invoke-Checked $VenvPython @('-m', 'streamlit', 'run', 'app\streamlit_app.py') }
    }
}
finally {
    Pop-Location
}
