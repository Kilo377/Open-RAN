param(
    [string]$Distribution = "Ubuntu"
)

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    Write-Error "Run this script from an elevated PowerShell window. WSL installation requires Administrator privileges."
    exit 1
}

wsl --install -d $Distribution

Write-Host ""
Write-Host "If Windows asks for a restart, restart first. Then open the installed $Distribution terminal and install:"
Write-Host "  sudo apt update"
Write-Host "  sudo apt install -y python3.11 python3.11-venv python3-pip git"
Write-Host "  python3.11 -m venv ~/oran_sionna_gpu"
Write-Host "  source ~/oran_sionna_gpu/bin/activate"
Write-Host "  pip install --upgrade pip"
Write-Host "  pip install pyyaml numpy openai tensorflow[and-cuda] sionna"
Write-Host "  python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
