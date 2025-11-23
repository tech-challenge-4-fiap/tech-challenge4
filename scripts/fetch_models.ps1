<#
Download dos modelos da Release GitHub (ex: v1.0) para ./models
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\fetch_models.ps1 -Tag v1.0
  ou dentro do PowerShell: .\scripts\fetch_models.ps1 -Tag v1.0
#>

param(
    [string]$Tag = 'v1.0',
    [switch]$Clobber
)

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "gh CLI não encontrado. Instale: https://cli.github.com/ e autentique com 'gh auth login'"
    exit 1
}

$dest = Join-Path -Path (Get-Location) -ChildPath "models"

if (-not (Test-Path -Path $dest)) {
    New-Item -ItemType Directory -Path $dest | Out-Null
}

$downloadArgs = @($Tag, "--dir", $dest)
if ($Clobber) {
    $downloadArgs += "--clobber"
} else {
    $downloadArgs += "--skip-existing"
}

Write-Host "Baixando assets da release $Tag para $dest ..."
gh release download @downloadArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "Modelos baixados com sucesso em: $dest"
    Get-ChildItem $dest | Format-Table Name, Length
} else {
    Write-Error "Falha ao baixar release. Verifique se a tag '$Tag' existe e se você está autenticado no gh."
    exit $LASTEXITCODE
}

