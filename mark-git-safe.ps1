# Get the current directory and convert to Git-friendly format
$currentPath = (Get-Location).Path -replace '\\', '/'

# Add triple slashes for UNC/network compatibility if needed
if ($currentPath -like '//*') {
    $safePath = "///" + $currentPath.TrimStart('/')
} else {
    $safePath = $currentPath
}

# Mark it as a safe Git directory
git config --global --add safe.directory "$safePath"

Write-Host "Marked as safe Git directory:" $safePath

# Run by :# .\mark-git-safe.ps1