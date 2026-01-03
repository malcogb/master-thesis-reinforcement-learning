# Script pour arrêter tous les builds Unity

Write-Host "`n=== Arrêt des Builds Unity ===" -ForegroundColor Cyan

# Chercher tous les processus AeroPatrol_Drone
$processes = Get-Process -Name "AeroPatrol_Drone" -ErrorAction SilentlyContinue

if ($processes) {
    $count = $processes.Count
    Write-Host "⚠️  $count build(s) Unity trouvé(s)..." -ForegroundColor Yellow
    
    # Afficher les détails des builds
    foreach ($proc in $processes) {
        Write-Host "  - PID: $($proc.Id) | Nom: $($proc.ProcessName)" -ForegroundColor Gray
    }
    
    # Demander confirmation
    Write-Host "`nArrêt en cours..." -ForegroundColor Yellow
    Stop-Process -Name "AeroPatrol_Drone" -Force
    
    Write-Host "✅ $count build(s) arrêté(s) avec succès!`n" -ForegroundColor Green
} else {
    Write-Host "ℹ️  Aucun build Unity en cours d'exécution.`n" -ForegroundColor Cyan
}

