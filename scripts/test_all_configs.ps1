# PowerShell script to test all config files and compare results
# Usage: .\scripts\test_all_configs.ps1 <audio_file>

param(
    [Parameter(Mandatory=$true)]
    [string]$AudioFile
)

# Check if audio file exists
if (-not (Test-Path $AudioFile)) {
    Write-Host "Error: Audio file not found: $AudioFile" -ForegroundColor Red
    exit 1
}

# Get all config files
$configs = @(
    "minimal_mode",
    "balanced_mode", 
    "high_quality_mode"
)

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Testing audio file with all configurations" -ForegroundColor Cyan
Write-Host "Audio: $AudioFile" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$results = @()

foreach ($config in $configs) {
    Write-Host ""
    Write-Host "▶ Testing with config: $config" -ForegroundColor Yellow
    Write-Host "-" * 80 -ForegroundColor DarkGray
    
    # Run the test and capture output
    $output = uv run python -m src.utils.cli test $AudioFile --config $config 2>&1
    
    # Display the output
    Write-Output $output
    
    # Try to extract bitrate from output
    $bitrateMatch = $output | Select-String -Pattern "Total:.*-> ([\d.]+) bps"
    if ($bitrateMatch) {
        $bitrate = $bitrateMatch.Matches.Groups[1].Value
        $results += [PSCustomObject]@{
            Config = $config
            Bitrate = [float]$bitrate
        }
    }
    
    Write-Host ""
}

# Display summary
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "SUMMARY - Bitrate Comparison" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

if ($results.Count -gt 0) {
    $results | Sort-Object Bitrate | Format-Table -AutoSize
    
    Write-Host ""
    Write-Host "💡 Lower bitrate = better compression" -ForegroundColor Green
    Write-Host "💡 Reconstructed audio files saved with '_reconstructed.wav' suffix" -ForegroundColor Green
} else {
    Write-Host "No results captured. Please check the output above for errors." -ForegroundColor Yellow
}
