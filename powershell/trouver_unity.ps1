# Script PowerShell pour trouver automatiquement Unity.exe
# Usage: .\trouver_unity.ps1

Write-Host "🔍 Recherche de Unity.exe..." -ForegroundColor Cyan
Write-Host ""

# Chemins communs où Unity est installé
$searchPaths = @(
    "C:\Program Files\Unity\Hub\Editor",
    "C:\Program Files (x86)\Unity",
    "C:\Program Files\Unity",
    "$env:LOCALAPPDATA\Programs\Unity\Hub\Editor"
)

$foundPaths = @()

foreach ($basePath in $searchPaths) {
    if (Test-Path $basePath) {
        Write-Host "📂 Recherche dans : $basePath" -ForegroundColor Gray
        
        # Chercher Unity.exe récursivement (max 3 niveaux de profondeur)
        $unityExes = Get-ChildItem -Path $basePath -Recurse -Filter "Unity.exe" -ErrorAction SilentlyContinue -Depth 3
        
        if ($unityExes) {
            foreach ($unityExe in $unityExes) {
                $foundPaths += $unityExe.FullName
                Write-Host "   ✅ Trouvé : $($unityExe.FullName)" -ForegroundColor Green
            }
        }
    }
}

Write-Host ""

if ($foundPaths.Count -eq 0) {
    Write-Host "❌ Unity.exe non trouvé dans les chemins communs." -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Méthodes alternatives :" -ForegroundColor Yellow
    Write-Host "   1. Ouvrir Unity Hub → Installs → Show in Explorer" -ForegroundColor White
    Write-Host "   2. Ouvrir Unity → Gestionnaire des tâches → Clic droit Unity.exe → Ouvrir l'emplacement" -ForegroundColor White
    Write-Host "   3. Chercher manuellement dans C:\Program Files\Unity" -ForegroundColor White
}
elseif ($foundPaths.Count -eq 1) {
    Write-Host "✅ Chemin unique trouvé :" -ForegroundColor Green
    Write-Host ""
    Write-Host "   $($foundPaths[0])" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 Copiez ce chemin dans launch_unity_parallel.ps1 (ligne 8)" -ForegroundColor Cyan
}
else {
    Write-Host "✅ Plusieurs installations Unity trouvées :" -ForegroundColor Green
    Write-Host ""
    for ($i = 0; $i -lt $foundPaths.Count; $i++) {
        Write-Host "   [$($i+1)] $($foundPaths[$i])" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "💡 Choisissez le numéro de l'installation que vous voulez utiliser" -ForegroundColor Cyan
    Write-Host "   Puis copiez le chemin dans launch_unity_parallel.ps1 (ligne 8)" -ForegroundColor Cyan
}

Write-Host ""

