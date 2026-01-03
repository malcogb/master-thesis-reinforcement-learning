# Script PowerShell pour lancer plusieurs builds Unity pour l'entraînement parallèle
# Usage: .\launch_unity_builds.ps1
#
# PREREQUIS :
# - Avoir créé un build Unity dans le dossier Builds/
# - Le build doit accepter --port en argument de ligne de commande

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LANCEMENT BUILDS UNITY PARALLELES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# MODIFIER CE CHEMIN SELON VOTRE BUILD
$buildPath = "C:\Users\MSI\New_level\AI_Master\Memoire_Robotique\AeroPatrol_drone\Builds\AeroPatrol_Drone.exe"

# Vérifier que le build existe
if (-not (Test-Path $buildPath)) {
    Write-Host "ERREUR : Build Unity non trouve a :" -ForegroundColor Red
    Write-Host "  $buildPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "SOLUTION :" -ForegroundColor Yellow
    Write-Host "  1. Creez un build Unity (File > Build Settings > Build)" -ForegroundColor Cyan
    Write-Host "  2. Placez l'executable dans le dossier Builds/" -ForegroundColor Cyan
    Write-Host "  3. Modifiez le chemin dans ce script si necessaire" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Consultez : marl/guide_creer_builds_unity.md" -ForegroundColor Green
    exit 1
}

# Vérifier que le dossier Builds existe
$buildsDir = Split-Path -Parent $buildPath
if (-not (Test-Path $buildsDir)) {
    Write-Host "ERREUR : Dossier Builds non trouve : $buildsDir" -ForegroundColor Red
    exit 1
}

# Ports à utiliser (correspond à NUM_PARALLEL_ENVS dans config.py)
# IMPORTANT : Le script lit automatiquement NUM_PARALLEL_ENVS depuis config.py
# Si NUM_PARALLEL_ENVS = 2, utiliser 2 ports (9000, 9001)
# Si NUM_PARALLEL_ENVS = 4, utiliser 4 ports (9000, 9001, 9002, 9003)

# Lire NUM_PARALLEL_ENVS depuis config.py
$configPath = "python\config.py"
$numParallelEnvs = 4  # Valeur par défaut

if (Test-Path $configPath) {
    $configContent = Get-Content $configPath -Raw
    if ($configContent -match "NUM_PARALLEL_ENVS\s*=\s*(\d+)") {
        $numParallelEnvs = [int]$matches[1]
        Write-Host "✅ NUM_PARALLEL_ENVS détecté depuis config.py : $numParallelEnvs" -ForegroundColor Green
    } else {
        Write-Host "⚠️  NUM_PARALLEL_ENVS non trouvé dans config.py, utilisation de la valeur par défaut : $numParallelEnvs" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  config.py non trouvé, utilisation de la valeur par défaut : $numParallelEnvs" -ForegroundColor Yellow
}

# Générer les ports automatiquement (9000, 9001, ..., 9000+N-1)
$ports = @()
for ($i = 0; $i -lt $numParallelEnvs; $i++) {
    $ports += 9000 + $i
}
$numBuilds = $ports.Length

Write-Host "✅ Configuration : $numBuilds builds sur ports $($ports -join ', ')" -ForegroundColor Cyan
Write-Host ""

Write-Host "Configuration détaillée :" -ForegroundColor Cyan
Write-Host "  Build  : $buildPath"
Write-Host "  Ports  : $($ports -join ', ')"
Write-Host "  Nombre : $numBuilds builds paralleles (NUM_PARALLEL_ENVS = $numParallelEnvs)"
Write-Host ""

# Vérifier si des builds Unity sont déjà en cours
$existingBuilds = Get-Process -Name "AeroPatrol_Drone" -ErrorAction SilentlyContinue
if ($existingBuilds) {
    Write-Host "ATTENTION : Des builds Unity sont deja en cours :" -ForegroundColor Yellow
    foreach ($proc in $existingBuilds) {
        Write-Host "  - Processus (PID: $($proc.Id))" -ForegroundColor Yellow
    }
    Write-Host ""
    
    $response = Read-Host "Voulez-vous fermer les builds existants ? (O/N)"
    if ($response -eq "O" -or $response -eq "o") {
        Write-Host "Fermeture des builds existants..." -ForegroundColor Yellow
        Stop-Process -Name "AeroPatrol_Drone" -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Write-Host "Builds fermes." -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "Les builds existants seront conserves." -ForegroundColor Yellow
        Write-Host ""
    }
}

# Vérifier que les ports sont disponibles
Write-Host "Verification des ports..." -ForegroundColor Cyan
$portsInUse = @()
foreach ($port in $ports) {
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connection) {
        $portsInUse += $port
        Write-Host "  Port $port : OCCUPE" -ForegroundColor Red
    } else {
        Write-Host "  Port $port : Disponible" -ForegroundColor Green
    }
}

if ($portsInUse.Count -gt 0) {
    Write-Host ""
    Write-Host "ATTENTION : Certains ports sont deja utilises : $($portsInUse -join ', ')" -ForegroundColor Yellow
    Write-Host "Les builds risquent de ne pas pouvoir demarrer sur ces ports." -ForegroundColor Yellow
    Write-Host ""
    
    $response = Read-Host "Continuer quand meme ? (O/N)"
    if ($response -ne "O" -and $response -ne "o") {
        Write-Host "Annulation." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "Lancement des builds Unity..." -ForegroundColor Green
Write-Host ""

$buildProcesses = @()

# Lancer les builds Unity
foreach ($port in $ports) {
    Write-Host "Lancement du build Unity (port $port)..." -ForegroundColor Yellow
    
    try {
        # Lancer le build avec --port en argument
        $process = Start-Process -FilePath $buildPath -ArgumentList "--port", $port -PassThru -WindowStyle Normal
        
        if ($process) {
            $buildProcesses += @{
                Port = $port
                Process = $process
                PID = $process.Id
            }
            Write-Host "  Build lance (PID: $($process.Id), Port: $port)" -ForegroundColor Green
        } else {
            Write-Host "  Erreur : Impossible de lancer le build" -ForegroundColor Red
        }
        
        # Attendre un peu entre chaque lancement pour éviter les conflits
        Start-Sleep -Seconds 2
    }
    catch {
        Write-Host "  Erreur lors du lancement (port $port) : $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  BUILDS UNITY LANCES" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Afficher le résumé
Write-Host "Resume :" -ForegroundColor Cyan
foreach ($bp in $buildProcesses) {
    Write-Host "  Port $($bp.Port) : PID $($bp.PID)" -ForegroundColor White
}

Write-Host ""
Write-Host "ETAPES SUIVANTES :" -ForegroundColor Cyan
Write-Host "  1. Attendez que chaque build Unity demarre (10-30 secondes)" -ForegroundColor White
Write-Host "  2. Verifiez dans chaque fenetre Unity :" -ForegroundColor White
Write-Host "     [UnityComms] Port definit depuis argument : XXXX" -ForegroundColor Gray
Write-Host "     [UnityComms] Server started on port XXXX" -ForegroundColor Gray
Write-Host "  3. Lancez l'entrainement Python :" -ForegroundColor White
Write-Host "     cd marl" -ForegroundColor Gray
Write-Host "     python train_marl.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Pour arreter tous les builds :" -ForegroundColor Yellow
Write-Host "  Stop-Process -Name 'AeroPatrol_Drone' -Force" -ForegroundColor Gray
Write-Host ""

# Attendre un peu pour voir si les builds démarrent correctement
Write-Host "Attente de 5 secondes pour verifier le demarrage..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Vérifier que les processus sont toujours actifs
$activeBuilds = Get-Process -Name "AeroPatrol_Drone" -ErrorAction SilentlyContinue
if ($activeBuilds) {
    Write-Host "  $($activeBuilds.Count) build(s) actif(s)" -ForegroundColor Green
} else {
    Write-Host "  ATTENTION : Aucun build actif detecte" -ForegroundColor Red
    Write-Host "  Verifiez les logs Unity pour les erreurs" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Les builds Unity sont prets pour l'entrainement parallele !" -ForegroundColor Green


