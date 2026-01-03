# Script PowerShell pour lancer plusieurs instances Unity pour l'entraînement parallèle
# Usage: .\launch_unity_parallel.ps1
#
# IMPORTANT : Unity Editor ne permet PAS d'ouvrir plusieurs instances du même projet
# Utilisez plutôt des BUILDS Unity (voir guide_creer_builds_unity.md)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LANCEMENT INSTANCES UNITY PARALLELES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Détecter les instances Unity existantes
$unityProcesses = Get-Process -Name "Unity" -ErrorAction SilentlyContinue
if ($unityProcesses) {
    Write-Host "ATTENTION : Des instances Unity sont deja ouvertes :" -ForegroundColor Yellow
    foreach ($proc in $unityProcesses) {
        Write-Host "  - Processus Unity (PID: $($proc.Id))" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Unity Editor ne permet PAS d'ouvrir plusieurs instances du meme projet." -ForegroundColor Red
    Write-Host ""
    Write-Host "SOLUTIONS :" -ForegroundColor Cyan
    Write-Host "  1. RECOMMANDE : Utiliser des BUILDS Unity (exécutables)" -ForegroundColor Green
    Write-Host "     Consultez : marl/guide_creer_builds_unity.md" -ForegroundColor Gray
    Write-Host "     Script    : launch_unity_builds.ps1" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. ALTERNATIVE : Fermer toutes les instances Unity et lancer manuellement" -ForegroundColor Yellow
    Write-Host "     - Fermez toutes les fenetres Unity" -ForegroundColor Gray
    Write-Host "     - Ouvrez Unity manuellement pour chaque instance" -ForegroundColor Gray
    Write-Host "     - Configurez le port dans chaque instance (9000, 9001, 9002, 9003)" -ForegroundColor Gray
    Write-Host ""
    
    $response = Read-Host "Voulez-vous fermer les instances Unity existantes ? (O/N)"
    if ($response -eq "O" -or $response -eq "o") {
        Write-Host "Fermeture des instances Unity..." -ForegroundColor Yellow
        Stop-Process -Name "Unity" -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Write-Host "Instances Unity fermees." -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "Annulation. Utilisez des builds Unity pour l'entrainement parallele." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "LIMITATION UNITY EDITOR :" -ForegroundColor Yellow
Write-Host "Unity Editor ne permet PAS d'ouvrir plusieurs instances du meme projet." -ForegroundColor Red
Write-Host ""
Write-Host "RECOMMANDATION : Utilisez des BUILDS Unity (exécutables)" -ForegroundColor Green
Write-Host "  - Consultez : marl/guide_creer_builds_unity.md" -ForegroundColor Cyan
Write-Host "  - Script    : launch_unity_builds.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Si vous voulez quand meme lancer l'editeur Unity :" -ForegroundColor Yellow
Write-Host "  Vous devez ouvrir manuellement chaque instance Unity" -ForegroundColor Gray
Write-Host "  et configurer le port dans UnityComms (9000, 9001, 9002, 9003)" -ForegroundColor Gray
Write-Host ""

$continue = Read-Host "Continuer avec l'editeur Unity ? (O/N)"
if ($continue -ne "O" -and $continue -ne "o") {
    Write-Host "Annulation. Utilisez des builds Unity pour l'entrainement parallele." -ForegroundColor Yellow
    exit 0
}

# MODIFIER CES CHEMINS SELON VOTRE INSTALLATION
$unityPath = "C:\Program Files\Unity\Hub\Editor\2021.3.45f2\Editor\Unity.exe"
$projectPath = "C:\Users\MSI\New_level\AI_Master\Memoire_Robotique\AeroPatrol_drone"

# Vérifier que les chemins existent
if (-not (Test-Path $unityPath)) {
    Write-Host "Erreur : Unity.exe non trouve a : $unityPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $projectPath)) {
    Write-Host "Erreur : Projet Unity non trouve a : $projectPath" -ForegroundColor Red
    exit 1
}

$ports = @(9000, 9001, 9002, 9003)

Write-Host ""
Write-Host "Configuration :" -ForegroundColor Cyan
Write-Host "  Unity  : $unityPath"
Write-Host "  Projet : $projectPath"
Write-Host "  Ports  : $($ports -join ', ')"
Write-Host ""
Write-Host "ATTENTION : Unity ne lancera probablement qu'une seule instance." -ForegroundColor Yellow
Write-Host ""

# Essayer de lancer une seule instance (Unity ne permettra pas plusieurs instances)
Write-Host "Tentative de lancement d'une instance Unity..." -ForegroundColor Yellow

try {
    Start-Process -FilePath $unityPath -ArgumentList "-projectPath `"$projectPath`""
    Start-Sleep -Seconds 3
    Write-Host "  Instance Unity lancee" -ForegroundColor Green
}
catch {
    Write-Host "  Erreur lors du lancement : $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "INSTRUCTIONS MANUELLES :" -ForegroundColor Cyan
Write-Host "  1. Ouvrez manuellement 4 instances Unity (une par une)" -ForegroundColor White
Write-Host "  2. Dans chaque instance, configurez le port dans UnityComms :" -ForegroundColor White
Write-Host "     - Instance 1 : Port 9000" -ForegroundColor Gray
Write-Host "     - Instance 2 : Port 9001" -ForegroundColor Gray
Write-Host "     - Instance 3 : Port 9002" -ForegroundColor Gray
Write-Host "     - Instance 4 : Port 9003" -ForegroundColor Gray
Write-Host "  3. Lancez chaque instance en mode Play (Ctrl+P)" -ForegroundColor White
Write-Host "  4. Verifiez dans chaque console Unity : 'Server started on port XXXX'" -ForegroundColor White
Write-Host "  5. Lancez l'entrainement Python : cd marl && python train_marl.py" -ForegroundColor White
Write-Host ""
Write-Host "RECOMMANDATION : Utilisez des builds Unity pour automatiser tout cela." -ForegroundColor Green
Write-Host "  Consultez : marl/guide_creer_builds_unity.md" -ForegroundColor Cyan
