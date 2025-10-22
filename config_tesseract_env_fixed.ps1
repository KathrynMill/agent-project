#!/usr/bin/env powershell
# Tesseract OCR Environment Configuration Script

Write-Host "Starting Tesseract OCR environment configuration..."

# Set Tesseract installation path
$tesseractPath = "C:\Program Files\Tesseract-OCR"
$tessdataPath = "$tesseractPath\tessdata"

# Check if Tesseract installation exists
if (-not (Test-Path $tesseractPath)) {
    Write-Host "Error: Tesseract installation directory not found at $tesseractPath"
    Exit 1
}

# Check if tessdata directory exists
if (-not (Test-Path $tessdataPath)) {
    Write-Host "Error: tessdata directory not found at $tessdataPath"
    Exit 1
}

# Add Tesseract path to system PATH environment variable
try {
    # Get current system PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    
    # Check if Tesseract path is already included
    if (-not ($currentPath -like "*$tesseractPath*")) {
        # Add path
        $newPath = "$currentPath;$tesseractPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        Write-Host "Successfully added Tesseract path to system PATH"
    } else {
        Write-Host "Tesseract path is already in system PATH"
    }
} catch {
    Write-Host "Failed to set PATH environment variable: $_"
}

# Set TESSDATA_PREFIX environment variable
try {
    # Get current TESSDATA_PREFIX
    $currentTessdataPrefix = [Environment]::GetEnvironmentVariable("TESSDATA_PREFIX", "Machine")
    
    # Check if already set
    if (-not $currentTessdataPrefix) {
        # Set environment variable
        [Environment]::SetEnvironmentVariable("TESSDATA_PREFIX", $tessdataPath, "Machine")
        Write-Host "Successfully set TESSDATA_PREFIX environment variable to $tessdataPath"
    } else {
        Write-Host "TESSDATA_PREFIX environment variable is already set to $currentTessdataPrefix"
    }
} catch {
    Write-Host "Failed to set TESSDATA_PREFIX environment variable: $_"
}

Write-Host ""
Write-Host "IMPORTANT NOTE:"
Write-Host "1. Environment variable configuration is complete"
Write-Host "2. You MUST restart command prompt or IDE for changes to take effect"
Write-Host "3. After restart, verify installation with: tesseract --version"
Write-Host ""
Write-Host "Configuration script execution complete!"