# Script pour vérifier l'état des builds Unity
# Vérifie si les processus Unity sont actifs et sur quels ports ils écoutent

Write-Host "=== Vérification des builds Unity ===" -ForegroundColor Cyan

# Ports utilisés
$ports = @(9001, 9002)

# Vérifier les processus Unity
$unityProcesses = Get-Process | Where-Object { $_.ProcessName -like "*Unity*" -or $_.ProcessName -like "*AeroPatrol*" }

if ($unityProcesses) {
    Write-Host "Processus Unity détectés :" -ForegroundColor Green
    foreach ($proc in $unityProcesses) {
        Write-Host "   - $($proc.ProcessName) (PID: $($proc.Id))" -ForegroundColor Yellow
    }
} else {
    Write-Host "Aucun processus Unity détecté !" -ForegroundColor Red
    Write-Host "   -> Les builds Unity ne sont probablement pas lancés" -ForegroundColor Yellow
}

Write-Host "`n=== Vérification des ports TCP ===" -ForegroundColor Cyan

# Vérifier les ports TCP en écoute
foreach ($port in $ports) {
    $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -InformationLevel Quiet
    if ($connection) {
        Write-Host "Port $port : ACTIF (en écoute)" -ForegroundColor Green
    } else {
        Write-Host "Port $port : INACTIF (pas d'écoute)" -ForegroundColor Red
    }
}

Write-Host "`nSi les ports sont inactifs, relancez les builds Unity avec :" -ForegroundColor Cyan
Write-Host "   .\launch_unity_builds.ps1" -ForegroundColor Yellow
