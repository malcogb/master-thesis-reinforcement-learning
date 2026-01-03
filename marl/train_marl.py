# train_marl.py
import os
import sys
import shutil
import json
import datetime
import random
import argparse

# Ajouter le répertoire parent au PYTHONPATH pour permettre les imports
# Cela permet de lancer le script depuis n'importe quel répertoire
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from tensorboard import program
import threading
import time
from python.aero_patrol_wrapper import AeroPatrolWrapper
from python.config import (
    SAVE_DIR, MODEL_NAME, EPISODES, MAX_STEPS_PER_EPISODE,
    LEARNING_RATE, GAMMA, TENSORBOARD_LOG_DIR, NUM_DRONES,
    NUM_PARALLEL_ENVS, USE_VECENV, USE_SUBPROC_VECENV, UNITY_PORT,
    CURRICULUM_ENABLED, CURRICULUM_START_STAGE,
    CURRICULUM_STAGE0_SUCCESS_RATE_THRESHOLD, CURRICULUM_STAGE0_MIN_EPISODES, CURRICULUM_STAGE0_MIN_TIMESTEPS,
    CURRICULUM_STAGE0_DETECTION_RATE_THRESHOLD, CURRICULUM_STAGE0_TRACKING_RATE_THRESHOLD, CURRICULUM_STAGE0_STABILITY_THRESHOLD,
    CURRICULUM_STAGE1_SUCCESS_RATE_THRESHOLD, CURRICULUM_STAGE1_MIN_EPISODES, CURRICULUM_STAGE1_MIN_TIMESTEPS,
    CURRICULUM_STAGE1_DETECTION_RATE_THRESHOLD, CURRICULUM_STAGE1_TRACKING_RATE_THRESHOLD, CURRICULUM_STAGE1_STABILITY_THRESHOLD,
    CHECKPOINT_SAVE_FREQ, CHECKPOINT_DIR
)
from python.helpers import ensure_dir
import numpy as np


class TensorBoardCallback(BaseCallback):
    """
    Callback personnalisé pour logger les métriques détaillées dans TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_info = []
        
    def _on_step(self) -> bool:
        # Les métriques standard sont déjà loggées par Stable-Baselines3
        # Ce callback permet d'ajouter des métriques personnalisées
        
        # Récupérer les infos de l'environnement si disponibles
        infos = self.locals.get('infos', [])
        if len(infos) > 0 and infos[0] is not None:
            info = infos[0]
            
            # Logger les métriques de récompenses détaillées
            if 'coverage_reward' in info:
                self.logger.record('rewards/coverage', info['coverage_reward'])
            if 'detection_reward' in info:
                self.logger.record('rewards/detection', info['detection_reward'])
            if 'proximity_reward' in info:
                self.logger.record('rewards/proximity', info['proximity_reward'])  # Nouvelle métrique
            if 'tracking_reward' in info:
                self.logger.record('rewards/tracking', info['tracking_reward'])
            if 'capture_reward' in info:
                self.logger.record('rewards/capture', info['capture_reward'])
            if 'central_alert_reward' in info:
                self.logger.record('rewards/central_alert', info['central_alert_reward'])
            
            # Logger les pénalités
            if 'collision_penalty' in info:
                self.logger.record('penalties/collision', info['collision_penalty'])
            if 'overlap_penalty' in info:
                self.logger.record('penalties/overlap', info['overlap_penalty'])
            if 'out_of_zone_penalty' in info:
                self.logger.record('penalties/out_of_zone', info['out_of_zone_penalty'])
            if 'obstacle_collision_penalty' in info:
                self.logger.record('penalties/obstacle_collision', info['obstacle_collision_penalty'])
            
            # Logger les métriques de performance
            if 'min_drone_distance' in info:
                self.logger.record('metrics/min_drone_distance', info['min_drone_distance'])
            if 'avg_drone_distance' in info:
                self.logger.record('metrics/avg_drone_distance', info['avg_drone_distance'])
            if 'min_distance_to_intruder' in info:
                self.logger.record('metrics/min_distance_to_intruder', info['min_distance_to_intruder'])  # Nouvelle métrique de diagnostic
            if 'coverage_ratio' in info:
                self.logger.record('metrics/coverage_ratio', info['coverage_ratio'])  # Nouvelle métrique de diagnostic
            if 'total_coverage_cells' in info:
                self.logger.record('metrics/total_coverage_cells', info['total_coverage_cells'])  # Nouvelle métrique de diagnostic
            if 'new_coverage_cells' in info:
                self.logger.record('metrics/new_coverage_cells', info['new_coverage_cells'])  # Nouvelle métrique de diagnostic
            if 'drones_out_of_zone' in info:
                self.logger.record('metrics/drones_out_of_zone', info['drones_out_of_zone'])
            if 'drones_near_obstacles' in info:
                self.logger.record('metrics/drones_near_obstacles', info['drones_near_obstacles'])
            if 'collision_count' in info:
                self.logger.record('metrics/collision_count', info['collision_count'])  # Nouvelle métrique
            if 'too_close_count' in info:
                self.logger.record('metrics/too_close_count', info['too_close_count'])  # Nouvelle métrique
            
            # Logger les métriques de curriculum learning
            if 'curriculum_stage' in info:
                self.logger.record('curriculum/stage', info['curriculum_stage'])
            if 'curriculum_episode_count' in info:
                self.logger.record('curriculum/episode_count', info['curriculum_episode_count'])
            if 'curriculum_episode_count_stage' in info:
                self.logger.record('curriculum/episode_count_stage', info['curriculum_episode_count_stage'])
            if 'curriculum_timesteps_count_stage' in info:
                self.logger.record('curriculum/timesteps_count_stage', info['curriculum_timesteps_count_stage'])
            if 'curriculum_success_rate' in info:
                self.logger.record('curriculum/success_rate', info['curriculum_success_rate'])
            if 'curriculum_detection_rate' in info:
                self.logger.record('curriculum/detection_rate', info['curriculum_detection_rate'])
            if 'curriculum_tracking_rate' in info:
                self.logger.record('curriculum/tracking_rate', info['curriculum_tracking_rate'])
            if 'curriculum_stability_cv' in info:
                self.logger.record('curriculum/stability_cv', info['curriculum_stability_cv'])
            
            # Logger la configuration du stage (pour voir ce qui est activé)
            if 'stage_enable_coverage' in info:
                self.logger.record('stage_config/enable_coverage', info['stage_enable_coverage'])
            if 'stage_enable_obstacles' in info:
                self.logger.record('stage_config/enable_obstacles', info['stage_enable_obstacles'])
            if 'stage_enable_zone' in info:
                self.logger.record('stage_config/enable_zone', info['stage_enable_zone'])
            if 'stage_enable_separation' in info:
                self.logger.record('stage_config/enable_separation', info['stage_enable_separation'])
            if 'stage_enable_central_alert' in info:
                self.logger.record('stage_config/enable_central_alert', info['stage_enable_central_alert'])
            if 'stage_enable_distance_penalty' in info:
                self.logger.record('stage_config/enable_distance_penalty', info['stage_enable_distance_penalty'])
            if 'stage_intruder_speed_mult' in info:
                self.logger.record('stage_config/intruder_speed_mult', info['stage_intruder_speed_mult'])
            if 'stage_radius_mult' in info:
                self.logger.record('stage_config/radius_mult', info['stage_radius_mult'])
        
        return True


def start_tensorboard(log_dir, port=6006):
    """
    Démarre TensorBoard dans un thread séparé.
    """
    def run_tensorboard():
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port)])
        url = tb.launch()
        print(f"📊 TensorBoard démarré sur {url}")
    
    thread = threading.Thread(target=run_tensorboard, daemon=True)
    thread.start()
    time.sleep(2)  # Attendre que TensorBoard démarre
    return thread


def export_config_to_dict():
    """
    Exporte toute la configuration depuis config.py vers un dictionnaire.
    """
    import python.config as config_module
    import inspect
    
    config_dict = {}
    
    # Récupérer tous les attributs de config qui ne sont pas des méthodes ou privés
    for attr_name in dir(config_module):
        if not attr_name.startswith('_'):
            try:
                attr_value = getattr(config_module, attr_name)
                # Ignorer les fonctions, méthodes et modules
                if not callable(attr_value) and not inspect.ismodule(attr_value):
                    # Convertir les types en JSON-serializable
                    if isinstance(attr_value, (int, float, str, bool, type(None))):
                        config_dict[attr_name] = attr_value
                    elif isinstance(attr_value, (list, tuple)):
                        config_dict[attr_name] = list(attr_value) if isinstance(attr_value, tuple) else attr_value
                    elif isinstance(attr_value, dict):
                        config_dict[attr_name] = attr_value
            except Exception:
                # Ignorer les attributs qui ne peuvent pas être récupérés
                pass
    
    return config_dict

def archive_experiment_files(experiment_dir, timestamp):
    """
    Archive les fichiers Python importants dans le dossier d'expérimentation.
    """
    files_to_archive = [
        # Fichiers Python principaux
        os.path.join(parent_dir, "python", "config.py"),
        os.path.join(parent_dir, "python", "aero_patrol_wrapper.py"),
        os.path.join(parent_dir, "python", "env_manager.py"),
        os.path.join(current_dir, "train_marl.py"),
        os.path.join(current_dir, "evaluate_marl.py"),
    ]
    
    # Fichiers Unity (scripts C#) - vérifier si le dossier Assets existe
    unity_scripts = [
        os.path.join(parent_dir, "Assets", "Scripts", "EnvManager.cs"),
        os.path.join(parent_dir, "Assets", "Scripts", "IntruderAgent.cs"),
        os.path.join(parent_dir, "Assets", "Scripts", "DroneAgent.cs"),
        os.path.join(parent_dir, "Assets", "Scripts", "PatrolZone.cs"),
        os.path.join(parent_dir, "Assets", "Scripts", "ObstacleManager.cs"),
        os.path.join(parent_dir, "Assets", "PeacefulPie", "Scripts", "UnityComms.cs"),
    ]
    
    # Ajouter les fichiers Unity seulement s'ils existent
    for unity_file in unity_scripts:
        if os.path.exists(unity_file):
            files_to_archive.append(unity_file)
    
    archived_files = []
    for file_path in files_to_archive:
        if os.path.exists(file_path):
            try:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(experiment_dir, file_name)
                shutil.copy2(file_path, dest_path)
                archived_files.append(file_name)
                print(f"📦 Fichier archivé : {file_name}")
            except Exception as e:
                print(f"⚠️  Impossible d'archiver {file_path} : {e}")
        else:
            # Afficher un avertissement seulement pour les fichiers Unity (optionnels)
            if "Assets" in file_path:
                pass  # Les fichiers Unity sont optionnels
            else:
                print(f"⚠️  Fichier non trouvé : {file_path}")
    
    return archived_files

def generate_analysis_report(experiment_dir, timestamp, env):
    """
    Génère un rapport d'analyse daté en référence au run.
    """
    report_filename = f"rapport_analyse_{timestamp}.md"
    report_path = os.path.join(experiment_dir, report_filename)
    
    # Récupérer les informations du curriculum depuis l'environnement
    curriculum_info = {}
    if hasattr(env, 'envs') and len(env.envs) > 0:
        wrapped_env = env.envs[0]
        if hasattr(wrapped_env, 'env'):
            aero_env = wrapped_env.env
            if hasattr(aero_env, 'current_stage'):
                curriculum_info = {
                    'current_stage': aero_env.current_stage,
                    'episode_count': aero_env.episode_count,
                    'episode_count_per_stage': aero_env.episode_count_per_stage.copy() if hasattr(aero_env, 'episode_count_per_stage') else {},
                    'timesteps_count_per_stage': aero_env.timesteps_count_per_stage.copy() if hasattr(aero_env, 'timesteps_count_per_stage') else {},
                    'recent_episodes_success': len(aero_env.recent_episodes_success) if hasattr(aero_env, 'recent_episodes_success') else 0,
                    'recent_episodes_detection': len(aero_env.recent_episodes_detection) if hasattr(aero_env, 'recent_episodes_detection') else 0,
                    'recent_episodes_tracking': len(aero_env.recent_episodes_tracking) if hasattr(aero_env, 'recent_episodes_tracking') else 0,
                }
                
                # Calculer les success rates si disponibles
                if hasattr(aero_env, 'recent_episodes_success') and len(aero_env.recent_episodes_success) > 0:
                    curriculum_info['success_rate'] = float(np.mean(aero_env.recent_episodes_success))
                    curriculum_info['detection_rate'] = float(np.mean(aero_env.recent_episodes_detection))
                    curriculum_info['tracking_rate'] = float(np.mean(aero_env.recent_episodes_tracking))
    
    # Générer le rapport
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Rapport d'Analyse - Run {timestamp}\n\n")
        f.write(f"**Date de génération** : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Configuration de l'Expérimentation\n\n")
        f.write(f"- **Run ID** : `{timestamp}`\n")
        f.write(f"- **Épisodes** : {EPISODES}\n")
        f.write(f"- **Steps par épisode** : {MAX_STEPS_PER_EPISODE}\n")
        f.write(f"- **Total timesteps** : {EPISODES * MAX_STEPS_PER_EPISODE}\n")
        f.write(f"- **Learning rate** : {LEARNING_RATE}\n")
        f.write(f"- **Gamma** : {GAMMA}\n")
        f.write(f"- **Nombre de drones** : {NUM_DRONES}\n\n")
        
        f.write("## 2. Curriculum Learning\n\n")
        f.write(f"- **Curriculum activé** : {CURRICULUM_ENABLED}\n")
        f.write(f"- **Stage de départ** : {CURRICULUM_START_STAGE}\n\n")
        
        if curriculum_info:
            f.write("### État Final du Curriculum\n\n")
            f.write(f"- **Stage actuel** : {curriculum_info.get('current_stage', 'N/A')}\n")
            f.write(f"- **Épisodes totaux** : {curriculum_info.get('episode_count', 0)}\n")
            f.write(f"- **Épisodes par stage** :\n")
            for stage, count in curriculum_info.get('episode_count_per_stage', {}).items():
                f.write(f"  - Stage {stage} : {count} épisodes\n")
            f.write(f"- **Timesteps par stage** :\n")
            for stage, count in curriculum_info.get('timesteps_count_per_stage', {}).items():
                f.write(f"  - Stage {stage} : {count} timesteps\n")
            
            if 'success_rate' in curriculum_info:
                f.write(f"\n### Métriques de Progression (Derniers {curriculum_info.get('recent_episodes_success', 0)} épisodes)\n\n")
                f.write(f"- **Success Rate (Capture)** : {curriculum_info['success_rate']:.2%}\n")
                f.write(f"- **Detection Rate** : {curriculum_info['detection_rate']:.2%}\n")
                f.write(f"- **Tracking Rate** : {curriculum_info['tracking_rate']:.2%}\n")
        
        f.write("\n### Critères de Progression\n\n")
        f.write("#### Stage 0 → Stage 1\n")
        f.write(f"- Success Rate >= {CURRICULUM_STAGE0_SUCCESS_RATE_THRESHOLD:.0%}\n")
        f.write(f"- Detection Rate >= {CURRICULUM_STAGE0_DETECTION_RATE_THRESHOLD:.0%}\n")
        f.write(f"- Tracking Rate >= {CURRICULUM_STAGE0_TRACKING_RATE_THRESHOLD:.0%}\n")
        f.write(f"- Stabilité (CV) <= {CURRICULUM_STAGE0_STABILITY_THRESHOLD:.0%}\n")
        f.write(f"- Minimum épisodes : {CURRICULUM_STAGE0_MIN_EPISODES}\n")
        f.write(f"- Minimum timesteps : {CURRICULUM_STAGE0_MIN_TIMESTEPS}\n\n")
        
        f.write("#### Stage 1 → Stage 2\n")
        f.write(f"- Success Rate >= {CURRICULUM_STAGE1_SUCCESS_RATE_THRESHOLD:.0%}\n")
        f.write(f"- Detection Rate >= {CURRICULUM_STAGE1_DETECTION_RATE_THRESHOLD:.0%}\n")
        f.write(f"- Tracking Rate >= {CURRICULUM_STAGE1_TRACKING_RATE_THRESHOLD:.0%}\n")
        f.write(f"- Stabilité (CV) <= {CURRICULUM_STAGE1_STABILITY_THRESHOLD:.0%}\n")
        f.write(f"- Minimum épisodes : {CURRICULUM_STAGE1_MIN_EPISODES}\n")
        f.write(f"- Minimum timesteps : {CURRICULUM_STAGE1_MIN_TIMESTEPS}\n\n")
        
        f.write("## 3. Configuration des Stages\n\n")
        f.write("### Stage 0: Focus Pursuit\n")
        f.write("- **Intrus vitesse** : 0.7x\n")
        f.write("- **Rayons** : 2.0x (détection, tracking, capture)\n")
        f.write("- **Coverage** : ❌ Désactivé\n")
        f.write("- **Obstacles** : ❌ Désactivé\n")
        f.write("- **Zone** : ❌ Désactivé\n")
        f.write("- **Séparation** : ❌ Désactivé\n")
        f.write("- **Central Alert** : ❌ Désactivé\n")
        f.write("- **Distance Penalty** : ✅ Activé\n\n")
        
        f.write("### Stage 1: Obstacles + Zone\n")
        f.write("- **Intrus vitesse** : 0.9x\n")
        f.write("- **Rayons** : 1.2x\n")
        f.write("- **Coverage** : ❌ Désactivé\n")
        f.write("- **Obstacles** : ✅ Activé\n")
        f.write("- **Zone** : ✅ Activé\n")
        f.write("- **Séparation** : ❌ Désactivé\n")
        f.write("- **Central Alert** : ✅ Activé\n")
        f.write("- **Distance Penalty** : ✅ Activé\n\n")
        
        f.write("### Stage 2: Complet\n")
        f.write("- **Intrus vitesse** : 1.0x\n")
        f.write("- **Rayons** : 1.0x\n")
        f.write("- **Coverage** : ✅ Activé\n")
        f.write("- **Obstacles** : ✅ Activé\n")
        f.write("- **Zone** : ✅ Activé\n")
        f.write("- **Séparation** : ✅ Activé\n")
        f.write("- **Central Alert** : ✅ Activé\n")
        f.write("- **Distance Penalty** : ✅ Activé\n\n")
        
        f.write("## 4. Ajustements Appliqués\n\n")
        f.write("### Modifications pour améliorer la détection au Stage 0\n")
        f.write("- **Radius Multiplier Stage 0** : Augmenté de 1.5x à **2.0x**\n")
        f.write("  - Rayon de détection : 25.0 × 2.0 = **50.0 unités** (au lieu de 37.5)\n")
        f.write("  - Objectif : Faciliter la détection (distance moyenne observée = 44.78)\n\n")
        f.write("- **Distance Penalty** : Augmenté de -0.01 à **-0.02** (x2)\n")
        f.write("  - Objectif : Encourager l'approche de l'intrus\n\n")
        f.write("- **Proximity Reward** : Multiplicateurs augmentés\n")
        f.write("  - Avant détection : 0.2 → **0.3** (x1.5)\n")
        f.write("  - Après détection : 0.3 → **0.45** (x1.5)\n")
        f.write("  - Max reward : 0.5 → **0.75** (x1.5)\n\n")
        
        f.write("## 5. Métriques TensorBoard\n\n")
        f.write("Les métriques suivantes sont disponibles dans TensorBoard :\n\n")
        f.write("### Curriculum\n")
        f.write("- `curriculum/stage` : Stage actuel\n")
        f.write("- `curriculum/episode_count` : Nombre total d'épisodes\n")
        f.write("- `curriculum/episode_count_stage` : Épisodes dans le stage actuel\n")
        f.write("- `curriculum/timesteps_count_stage` : Timesteps dans le stage actuel\n")
        f.write("- `curriculum/success_rate` : Taux de succès (capture)\n")
        f.write("- `curriculum/detection_rate` : Taux de détection\n")
        f.write("- `curriculum/tracking_rate` : Taux de tracking\n")
        f.write("- `curriculum/stability_cv` : Coefficient de variation (stabilité)\n\n")
        
        f.write("### Configuration du Stage\n")
        f.write("- `stage_config/enable_coverage` : Coverage activé (1) ou non (0)\n")
        f.write("- `stage_config/enable_obstacles` : Obstacles activés (1) ou non (0)\n")
        f.write("- `stage_config/enable_zone` : Zone activée (1) ou non (0)\n")
        f.write("- `stage_config/enable_separation` : Séparation activée (1) ou non (0)\n")
        f.write("- `stage_config/enable_central_alert` : Central Alert activé (1) ou non (0)\n")
        f.write("- `stage_config/enable_distance_penalty` : Distance Penalty activé (1) ou non (0)\n")
        f.write("- `stage_config/intruder_speed_mult` : Multiplicateur de vitesse de l'intrus\n")
        f.write("- `stage_config/radius_mult` : Multiplicateur des rayons\n\n")
        
        f.write("## 6. Recommandations\n\n")
        f.write("1. **Surveiller les métriques de curriculum** dans TensorBoard pour suivre la progression\n")
        f.write("2. **Vérifier les success rates** pour chaque stage avant la progression\n")
        f.write("3. **Analyser la stabilité** (CV) pour s'assurer d'un apprentissage stable\n")
        f.write("4. **Comparer les performances** entre les stages pour évaluer l'efficacité du curriculum\n\n")
        
        f.write("---\n\n")
        f.write(f"*Rapport généré automatiquement le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"📊 Rapport d'analyse généré : {report_path}")
    return report_path


def save_experiment_info(experiment_dir, config_dict, model_info, timestamp):
    """
    Sauvegarde les informations de l'expérimentation dans un fichier JSON.
    """
    experiment_info = {
        "timestamp": timestamp,
        "experiment_date": datetime.datetime.now().isoformat(),
        "model_info": model_info,
        "config": config_dict,
        "system_info": {
            "python_version": sys.version.split()[0],  # Version Python seulement
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "device_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
        }
    }
    
    info_path = os.path.join(experiment_dir, "experiment_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    print(f"📝 Informations d'expérimentation sauvegardées : {info_path}")
    
    return info_path

def train(load_model_path=None, reset_curriculum=False):
    """
    Fonction principale d'entraînement.
    
    Args:
        load_model_path: Chemin vers un modèle à charger (ex: "models/ppo_marl_20251204_115027.zip")
                       Si None, crée un nouveau modèle
        reset_curriculum: Si True et load_model_path fourni, réinitialise le curriculum à zéro
                         Si False, continue le curriculum depuis l'état sauvegardé
    """
    # ======================================================================
    # 🎲 0. FIXATION DES GRAINES POUR REPRODUCTIBILITÉ
    # ======================================================================
    SEED = 42  # Graine fixe pour reproductibilité
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎲 Graines fixées pour reproductibilité (SEED={SEED})")
    
    # ======================================================================
    # 🧩 1. CRÉATION DU DOSSIER D'EXPÉRIMENTATION
    # ======================================================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(current_dir, "experiments", f"exp_{timestamp}")
    ensure_dir(experiment_dir)
    print(f"\n🚀 Dossier d'expérimentation créé : {experiment_dir}\n")
    
    # Vérifier la disponibilité du GPU
    # Note: PPO avec MLP (Multi-Layer Perceptron) n'est pas optimal sur GPU
    # Stable-Baselines3 recommande d'utiliser CPU pour les politiques MLP
    # Pour CNN, GPU est recommandé, mais nous utilisons MLP ici
    use_gpu_for_training = False  # Désactiver GPU pour PPO avec MLP (plus rapide sur CPU)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"✅ GPU détecté : {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"✅ CUDA version : {torch.version.cuda}")
        if use_gpu_for_training:
            print(f"✅ PyTorch utilisera le GPU pour l'entraînement")
        else:
            print(f"ℹ️  GPU disponible mais utilisation du CPU recommandée pour PPO avec MLP")
            print(f"   (Le GPU est plus lent pour les politiques MLP selon Stable-Baselines3)")
            device = "cpu"
    else:
        print("⚠️  Aucun GPU détecté. L'entraînement utilisera le CPU.")
        device = "cpu"
    
    if use_gpu_for_training and torch.cuda.is_available():
        device = "cuda"
    
    # ======================================================================
    # ⚙️ 2. EXPORT ET SAUVEGARDE DE LA CONFIGURATION
    # ======================================================================
    config_dict = export_config_to_dict()
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"📝 Configuration sauvegardée : {config_path}")
    
    # ======================================================================
    # 📦 3. ARCHIVAGE DES FICHIERS IMPORTANTS
    # ======================================================================
    print("\n📦 Archivage des fichiers importants...")
    archived_files = archive_experiment_files(experiment_dir, timestamp)
    print(f"✅ {len(archived_files)} fichiers archivés\n")
    
    # Crée les dossiers de sauvegarde si nécessaire
    ensure_dir(SAVE_DIR)
    ensure_dir(TENSORBOARD_LOG_DIR)
    
    # Crée un sous-dossier avec timestamp pour cette session d'entraînement
    session_log_dir = os.path.join(TENSORBOARD_LOG_DIR, f"ppo_marl_{timestamp}")
    ensure_dir(session_log_dir)
    
    # Lier le dossier de logs TensorBoard au dossier d'expérimentation
    # Créer un lien symbolique ou copier les logs dans experiment_dir
    experiment_logs_dir = os.path.join(experiment_dir, "tensorboard_logs")
    # On créera un fichier de référence plutôt qu'un lien (plus portable)
    logs_reference = {
        "tensorboard_log_dir": session_log_dir,
        "relative_path": os.path.relpath(session_log_dir, experiment_dir)
    }
    with open(os.path.join(experiment_dir, "logs_reference.json"), 'w') as f:
        json.dump(logs_reference, f, indent=2)
    
    print(f"📊 Logs TensorBoard : {session_log_dir}")
    
    # ======================================================================
    # 🌐 3.5. CRÉATION DES ENVIRONNEMENTS (SÉQUENTIEL OU PARALLÈLE)
    # ======================================================================
    print("🔍 [DIAGNOSTIC] Début de la création des environnements...")
    sys.stdout.flush()  # Forcer l'affichage immédiat
    
    if USE_VECENV and NUM_PARALLEL_ENVS > 1:
        vecenv_type = "SubprocVecEnv" if USE_SUBPROC_VECENV else "DummyVecEnv"
        print(f"🚀 Création de {NUM_PARALLEL_ENVS} environnements parallèles avec {vecenv_type}")
        sys.stdout.flush()
        
        def make_env(rank):
            """Crée un environnement avec un port spécifique (fonction picklable pour SubprocVecEnv)."""
            def _init():
                print(f"🔍 [DIAGNOSTIC] Création de l'environnement {rank} (port {UNITY_PORT + rank})...")
                sys.stdout.flush()
                try:
                    env = AeroPatrolWrapper(num_drones=NUM_DRONES, max_steps=MAX_STEPS_PER_EPISODE, port=UNITY_PORT + rank)
                    print(f"🔍 [DIAGNOSTIC] Environnement {rank} créé, reset en cours...")
                    sys.stdout.flush()
                    env.reset(seed=SEED + rank)  # Seed différent par environnement pour diversité
                    print(f"🔍 [DIAGNOSTIC] Environnement {rank} reset terminé")
                    sys.stdout.flush()
                    env = Monitor(env, os.path.join(session_log_dir, f"env_{rank}"), allow_early_resets=True)
                    return env
                except Exception as e:
                    print(f"❌ [DIAGNOSTIC] Erreur lors de la création de l'environnement {rank} : {e}")
                    sys.stdout.flush()
                    raise
            return _init
        
        # Créer les environnements avec des ports différents (9000, 9001, etc.)
        print(f"🔍 [DIAGNOSTIC] Création des fonctions d'environnement...")
        sys.stdout.flush()
        env_fns = [make_env(i) for i in range(NUM_PARALLEL_ENVS)]
        
        if USE_SUBPROC_VECENV:
            # Vraie parallélisation avec processus séparés
            print(f"🔍 [DIAGNOSTIC] Création de SubprocVecEnv...")
            sys.stdout.flush()
            env = SubprocVecEnv(env_fns, start_method='spawn')
            print(f"✅ {NUM_PARALLEL_ENVS} environnements créés en parallèle (SubprocVecEnv)")
            print(f"   → Ports : {UNITY_PORT} à {UNITY_PORT + NUM_PARALLEL_ENVS - 1}")
            print(f"   → Chaque environnement dans son propre processus Python")
            print(f"   ⚠️  Assurez-vous que {NUM_PARALLEL_ENVS} instances Unity sont lancées sur ces ports")
        else:
            # Traitement séquentiel mais batch (DummyVecEnv)
            print(f"🔍 [DIAGNOSTIC] Création de DummyVecEnv...")
            sys.stdout.flush()
            env = DummyVecEnv(env_fns)
            print(f"✅ {NUM_PARALLEL_ENVS} environnements créés (DummyVecEnv - séquentiel)")
            print(f"   → Ports : {UNITY_PORT} à {UNITY_PORT + NUM_PARALLEL_ENVS - 1}")
            print(f"   ⚠️  Assurez-vous que {NUM_PARALLEL_ENVS} instances Unity sont lancées sur ces ports")
    else:
        print("📦 Création d'un environnement unique (séquentiel)")
        sys.stdout.flush()
        print("🔍 [DIAGNOSTIC] Création de AeroPatrolWrapper...")
        sys.stdout.flush()
        # Initialise l'environnement MARL multi-drone
        env = AeroPatrolWrapper(num_drones=NUM_DRONES, max_steps=MAX_STEPS_PER_EPISODE)
        print("🔍 [DIAGNOSTIC] AeroPatrolWrapper créé, reset en cours...")
        sys.stdout.flush()
        # Fixer la graine de l'environnement pour reproductibilité
        env.reset(seed=SEED)
        print("🔍 [DIAGNOSTIC] Reset terminé, création du Monitor...")
        sys.stdout.flush()
        # Envelopper l'environnement avec Monitor pour logging
        env = Monitor(env, session_log_dir, allow_early_resets=True)
    
    print("🔍 [DIAGNOSTIC] Environnements créés avec succès")
    sys.stdout.flush()
    
    # ======================================================================
    # 🤖 4. CHARGEMENT OU CRÉATION DU MODÈLE
    # ======================================================================
    print("🔍 [DIAGNOSTIC] Début de la section chargement/création du modèle...")
    sys.stdout.flush()
    
    if load_model_path:
        # Normaliser le chemin du modèle
        if not os.path.isabs(load_model_path):
            # Chemin relatif : chercher dans models/
            models_dir = os.path.join(current_dir, SAVE_DIR)
            # Enlever "models/" du début si présent pour éviter duplication
            if load_model_path.startswith('models/'):
                load_model_path = load_model_path[7:]  # Enlever "models/"
            if not load_model_path.endswith('.zip'):
                load_model_path = f"{load_model_path}.zip"
            load_model_path = os.path.join(models_dir, load_model_path)
        
        if not os.path.exists(load_model_path):
            print(f"❌ Erreur : Modèle non trouvé : {load_model_path}")
            print(f"   Recherche dans : {os.path.dirname(load_model_path)}")
            return
        
        print(f"📦 Chargement du modèle : {os.path.basename(load_model_path)}")
        sys.stdout.flush()
        try:
            print("🔍 [DIAGNOSTIC] Début du chargement du modèle PPO...")
            sys.stdout.flush()
            model = PPO.load(load_model_path, env=env, device=device)
            print("✅ Modèle chargé avec succès")
            sys.stdout.flush()
            
            # ⚠️  IMPORTANT : Mettre à jour le tensorboard_log pour éviter les erreurs "No such file or directory"
            # Le modèle chargé peut avoir un ancien chemin TensorBoard qui n'existe plus
            try:
                model.tensorboard_log = session_log_dir
                if hasattr(model, 'logger'):
                    # Fermer l'ancien logger s'il existe
                    if hasattr(model.logger, 'close'):
                        try:
                            model.logger.close()
                        except:
                            pass
                    # Réinitialiser le logger avec le nouveau chemin
                    from stable_baselines3.common.logger import configure
                    model.logger = configure(session_log_dir, ["stdout", "csv", "tensorboard"])
                print(f"📊 TensorBoard log mis à jour : {session_log_dir}")
            except Exception as tb_error:
                print(f"⚠️  Avertissement : Impossible de mettre à jour TensorBoard log : {tb_error}")
                print(f"   → L'entraînement continuera mais peut échouer si l'ancien répertoire n'existe plus")
            
            # Option : Réinitialiser le curriculum si demandé
            if reset_curriculum:
                print("🔄 Réinitialisation du curriculum à zéro")
                if USE_VECENV and NUM_PARALLEL_ENVS > 1:
                    # VecEnv : reset tous les environnements
                    if USE_SUBPROC_VECENV:
                        # SubprocVecEnv : utiliser set_attr avec chemin d'attribut pour modifier les environnements dans les processus séparés
                        # Note: set_attr avec chemin d'attribut 'env.attribut' pour accéder à AeroPatrolWrapper via Monitor
                        try:
                            # Essayer d'utiliser set_attr avec chemin d'attribut
                            env.set_attr('env.current_stage', CURRICULUM_START_STAGE, indices=None)
                            env.set_attr('env.episode_count', 0, indices=None)
                            env.set_attr('env.episode_count_per_stage', {0: 0, 1: 0, 2: 0}, indices=None)
                            env.set_attr('env.timesteps_count_per_stage', {0: 0, 1: 0, 2: 0}, indices=None)
                            env.set_attr('env.recent_episodes_success', [], indices=None)
                            env.set_attr('env.recent_episodes_detection', [], indices=None)
                            env.set_attr('env.recent_episodes_tracking', [], indices=None)
                            env.set_attr('env.recent_episodes_rewards', [], indices=None)
                        except (AttributeError, TypeError) as e:
                            # Si set_attr avec chemin ne fonctionne pas, utiliser une méthode via call_method si disponible
                            # Sinon, les environnements seront reset au prochain épisode
                            print(f"   ⚠️  set_attr avec chemin d'attribut non supporté: {e}")
                            print("   ⚠️  Les environnements seront reset au prochain épisode (reset automatique)")
                            # Note: Les environnements seront automatiquement reset au prochain reset() avec le stage correct
                    else:
                        # DummyVecEnv : accès direct via env.envs
                        for i in range(NUM_PARALLEL_ENVS):
                            wrapped_env = env.envs[i]
                            if hasattr(wrapped_env, 'env'):
                                aero_env = wrapped_env.env
                                if hasattr(aero_env, 'current_stage'):
                                    aero_env.current_stage = CURRICULUM_START_STAGE
                                    aero_env.episode_count = 0
                                    aero_env.episode_count_per_stage = {0: 0, 1: 0, 2: 0}
                                    aero_env.timesteps_count_per_stage = {0: 0, 1: 0, 2: 0}
                                    aero_env.recent_episodes_success = []
                                    aero_env.recent_episodes_detection = []
                                    aero_env.recent_episodes_tracking = []
                                    aero_env.recent_episodes_rewards = []
                    print(f"   → Stage réinitialisé à {CURRICULUM_START_STAGE} pour tous les environnements")
                else:
                    # Environnement unique
                    env.env.current_stage = CURRICULUM_START_STAGE
                    env.env.episode_count = 0
                    env.env.episode_count_per_stage = {0: 0, 1: 0, 2: 0}
                    env.env.timesteps_count_per_stage = {0: 0, 1: 0, 2: 0}
                    env.env.recent_episodes_success = []
                    env.env.recent_episodes_detection = []
                    env.env.recent_episodes_tracking = []
                    env.env.recent_episodes_rewards = []
                    print(f"   → Stage réinitialisé à {CURRICULUM_START_STAGE}")
            else:
                # Tenter de restaurer l'état du curriculum depuis un fichier JSON
                curriculum_state_path = load_model_path.replace('.zip', '_curriculum_state.json')
                if os.path.exists(curriculum_state_path):
                    try:
                        with open(curriculum_state_path, 'r') as f:
                            curriculum_state = json.load(f)
                        
                        if USE_VECENV and NUM_PARALLEL_ENVS > 1:
                            # VecEnv : restaurer pour tous les environnements
                            if USE_SUBPROC_VECENV:
                                # SubprocVecEnv : utiliser set_attr avec chemin d'attribut pour restaurer l'état
                                # S'assurer que tous les dictionnaires ont toutes les clés (0, 1, 2)
                                default_episode_count = {0: 0, 1: 0, 2: 0}
                                default_timesteps_count = {0: 0, 1: 0, 2: 0}
                                episode_count_loaded = curriculum_state.get('episode_count_per_stage', default_episode_count)
                                timesteps_count_loaded = curriculum_state.get('timesteps_count_per_stage', default_timesteps_count)
                                
                                # Fusionner avec les valeurs par défaut pour garantir toutes les clés
                                episode_count_merged = {**default_episode_count, **episode_count_loaded}
                                timesteps_count_merged = {**default_timesteps_count, **timesteps_count_loaded}
                                
                                try:
                                    env.set_attr('env.current_stage', curriculum_state.get('current_stage', CURRICULUM_START_STAGE), indices=None)
                                    env.set_attr('env.episode_count', curriculum_state.get('episode_count', 0), indices=None)
                                    env.set_attr('env.episode_count_per_stage', episode_count_merged, indices=None)
                                    env.set_attr('env.timesteps_count_per_stage', timesteps_count_merged, indices=None)
                                    env.set_attr('env.recent_episodes_success', curriculum_state.get('recent_episodes_success', []), indices=None)
                                    env.set_attr('env.recent_episodes_detection', curriculum_state.get('recent_episodes_detection', []), indices=None)
                                    env.set_attr('env.recent_episodes_tracking', curriculum_state.get('recent_episodes_tracking', []), indices=None)
                                    env.set_attr('env.recent_episodes_rewards', curriculum_state.get('recent_episodes_rewards', []), indices=None)
                                except (AttributeError, TypeError) as e:
                                    print(f"   ⚠️  set_attr avec chemin d'attribut non supporté: {e}")
                                    print("   ⚠️  L'état du curriculum ne peut pas être restauré automatiquement avec SubprocVecEnv")
                                    print("   ℹ️  Les environnements utiliseront l'état par défaut (Stage 0)")
                            else:
                                # DummyVecEnv : accès direct via env.envs
                                for i in range(NUM_PARALLEL_ENVS):
                                    wrapped_env = env.envs[i]
                                    if hasattr(wrapped_env, 'env'):
                                        aero_env = wrapped_env.env
                                        if hasattr(aero_env, 'current_stage'):
                                            aero_env.current_stage = curriculum_state.get('current_stage', CURRICULUM_START_STAGE)
                                            aero_env.episode_count = curriculum_state.get('episode_count', 0)
                                            aero_env.episode_count_per_stage = curriculum_state.get('episode_count_per_stage', {0: 0, 1: 0, 2: 0})
                                            aero_env.timesteps_count_per_stage = curriculum_state.get('timesteps_count_per_stage', {0: 0, 1: 0, 2: 0})
                                            aero_env.recent_episodes_success = curriculum_state.get('recent_episodes_success', [])
                                            aero_env.recent_episodes_detection = curriculum_state.get('recent_episodes_detection', [])
                                            aero_env.recent_episodes_tracking = curriculum_state.get('recent_episodes_tracking', [])
                                            aero_env.recent_episodes_rewards = curriculum_state.get('recent_episodes_rewards', [])
                            print("✅ État du curriculum restauré depuis le fichier (tous les environnements)")
                            print(f"   → Stage actuel : {curriculum_state.get('current_stage', CURRICULUM_START_STAGE)}")
                            print(f"   → Épisodes totaux : {curriculum_state.get('episode_count', 0)}")
                        else:
                            # Environnement unique
                            env.env.current_stage = curriculum_state.get('current_stage', CURRICULUM_START_STAGE)
                            env.env.episode_count = curriculum_state.get('episode_count', 0)
                            
                            # S'assurer que tous les dictionnaires ont toutes les clés (0, 1, 2)
                            default_episode_count = {0: 0, 1: 0, 2: 0}
                            default_timesteps_count = {0: 0, 1: 0, 2: 0}
                            episode_count_loaded = curriculum_state.get('episode_count_per_stage', default_episode_count)
                            timesteps_count_loaded = curriculum_state.get('timesteps_count_per_stage', default_timesteps_count)
                            
                            # Fusionner avec les valeurs par défaut pour garantir toutes les clés
                            env.env.episode_count_per_stage = {**default_episode_count, **episode_count_loaded}
                            env.env.timesteps_count_per_stage = {**default_timesteps_count, **timesteps_count_loaded}
                            
                            env.env.recent_episodes_success = curriculum_state.get('recent_episodes_success', [])
                            env.env.recent_episodes_detection = curriculum_state.get('recent_episodes_detection', [])
                            env.env.recent_episodes_tracking = curriculum_state.get('recent_episodes_tracking', [])
                            env.env.recent_episodes_rewards = curriculum_state.get('recent_episodes_rewards', [])
                            print("✅ État du curriculum restauré depuis le fichier")
                            print(f"   → Stage actuel : {env.env.current_stage}")
                            print(f"   → Épisodes totaux : {env.env.episode_count}")
                            print(f"   → Épisodes par stage : {env.env.episode_count_per_stage}")
                            print(f"   → Timesteps par stage : {env.env.timesteps_count_per_stage}")
                    except Exception as e:
                        print(f"⚠️  Impossible de restaurer l'état du curriculum : {e}")
                        print("   → Utilisation de l'état par défaut")
                else:
                    print("ℹ️  Aucun fichier d'état du curriculum trouvé, utilisation de l'état par défaut")
                    if USE_VECENV and NUM_PARALLEL_ENVS > 1:
                        # Pour VecEnv, on ne peut pas accéder facilement au stage, donc on affiche juste un message
                        print(f"   → Stage par défaut : {CURRICULUM_START_STAGE} (pour tous les environnements)")
                    else:
                        print(f"   → Stage actuel : {env.env.current_stage}")
                        print(f"   → Épisodes totaux : {env.env.episode_count}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
            return
    else:
        # Créer un nouveau modèle
        print("🆕 Création d'un nouveau modèle PPO")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            verbose=1,
            tensorboard_log=session_log_dir,  # ✅ Activation de TensorBoard
            device=device,  # ✅ Utilisation explicite du GPU si disponible
            max_grad_norm=0.5,  # ✅ Gradient clipping pour stabiliser (Run 7: loss très élevée)
            seed=SEED,  # ✅ Graine pour reproductibilité
            ent_coef=0.01,  # ✅ Bonus d'entropie pour encourager l'exploration (évite convergence prématurée)
            # Hyperparamètres ajustés pour améliorer la stabilité (CV) et ep_rew_mean (2025-12-07)
            n_steps=3072,  # Augmenté de 2560 à 3072 (+20%) pour améliorer la stabilité et ep_rew_mean
            batch_size=96,  # Augmenté de 80 à 96 (+20%) pour gradients plus stables et meilleure convergence
            n_epochs=10,  # Augmenté de 8 à 10 (+25%) pour meilleure optimisation (était 12, réduit pour mises à jour plus fréquentes)
            gae_lambda=0.98,  # Conservé à 0.98 (meilleure estimation de la valeur)
            vf_coef=0.5,  # Augmenté de 0.5 (défaut) pour améliorer l'apprentissage de la value function (réduire value_loss)
        )
    
    # ======================================================================
    # 💾 5. CONFIGURATION DES CALLBACKS (TensorBoard + Checkpoint)
    # ======================================================================
    # Crée le callback TensorBoard pour métriques personnalisées
    tensorboard_callback = TensorBoardCallback()
    
    # Crée le callback de checkpoint pour sauvegarder périodiquement
    checkpoint_dir = os.path.join(current_dir, SAVE_DIR, CHECKPOINT_DIR)
    ensure_dir(checkpoint_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=checkpoint_dir,
        name_prefix="ppo_marl_checkpoint",
        save_replay_buffer=False,  # Ne pas sauvegarder le replay buffer (économise de l'espace)
        save_vecnormalize=False,  # Pas de VecNormalize utilisé
    )
    
    # 🎓 Callback personnalisé pour sauvegarder l'état du curriculum à chaque checkpoint
    class CurriculumStateCallback(BaseCallback):
        """Sauvegarde l'état du curriculum à chaque checkpoint."""
        def __init__(self, env, checkpoint_dir, verbose=0):
            super().__init__(verbose)
            self.env = env
            self.checkpoint_dir = checkpoint_dir
        
        def _get_curriculum_state(self):
            """Récupère l'état du curriculum depuis l'environnement (compatible VecEnv et environnement unique)."""
            # Vérifier si c'est un VecEnv (DummyVecEnv ou SubprocVecEnv)
            if hasattr(self.env, 'envs'):  # DummyVecEnv
                # Accéder au premier environnement (tous les environnements partagent le même état de curriculum)
                wrapped_env = self.env.envs[0]
                if hasattr(wrapped_env, 'env'):
                    aero_env = wrapped_env.env
                    if hasattr(aero_env, 'current_stage'):
                        return {
                            'current_stage': aero_env.current_stage,
                            'episode_count': aero_env.episode_count,
                            'episode_count_per_stage': aero_env.episode_count_per_stage.copy() if hasattr(aero_env, 'episode_count_per_stage') else {},
                            'timesteps_count_per_stage': aero_env.timesteps_count_per_stage.copy() if hasattr(aero_env, 'timesteps_count_per_stage') else {},
                            'recent_episodes_success': aero_env.recent_episodes_success.copy() if hasattr(aero_env, 'recent_episodes_success') else [],
                            'recent_episodes_detection': aero_env.recent_episodes_detection.copy() if hasattr(aero_env, 'recent_episodes_detection') else [],
                            'recent_episodes_tracking': aero_env.recent_episodes_tracking.copy() if hasattr(aero_env, 'recent_episodes_tracking') else [],
                            'recent_episodes_rewards': aero_env.recent_episodes_rewards.copy() if hasattr(aero_env, 'recent_episodes_rewards') else []
                        }
            elif hasattr(self.env, 'get_attr'):  # SubprocVecEnv
                # Utiliser get_attr pour accéder aux attributs dans les processus séparés
                try:
                    current_stage = self.env.get_attr('env.current_stage', indices=[0])[0]
                    episode_count = self.env.get_attr('env.episode_count', indices=[0])[0]
                    episode_count_per_stage = self.env.get_attr('env.episode_count_per_stage', indices=[0])[0]
                    timesteps_count_per_stage = self.env.get_attr('env.timesteps_count_per_stage', indices=[0])[0]
                    recent_episodes_success = self.env.get_attr('env.recent_episodes_success', indices=[0])[0]
                    recent_episodes_detection = self.env.get_attr('env.recent_episodes_detection', indices=[0])[0]
                    recent_episodes_tracking = self.env.get_attr('env.recent_episodes_tracking', indices=[0])[0]
                    recent_episodes_rewards = self.env.get_attr('env.recent_episodes_rewards', indices=[0])[0]
                    return {
                        'current_stage': current_stage,
                        'episode_count': episode_count,
                        'episode_count_per_stage': episode_count_per_stage.copy() if isinstance(episode_count_per_stage, dict) else {},
                        'timesteps_count_per_stage': timesteps_count_per_stage.copy() if isinstance(timesteps_count_per_stage, dict) else {},
                        'recent_episodes_success': recent_episodes_success.copy() if isinstance(recent_episodes_success, list) else [],
                        'recent_episodes_detection': recent_episodes_detection.copy() if isinstance(recent_episodes_detection, list) else [],
                        'recent_episodes_tracking': recent_episodes_tracking.copy() if isinstance(recent_episodes_tracking, list) else [],
                        'recent_episodes_rewards': recent_episodes_rewards.copy() if isinstance(recent_episodes_rewards, list) else []
                    }
                except Exception as e:
                    if self.verbose > 0:
                        print(f"⚠️  Impossible de récupérer l'état du curriculum depuis SubprocVecEnv : {e}")
                    return None
            else:  # Environnement unique
                if hasattr(self.env, 'env') and hasattr(self.env.env, 'current_stage'):
                    return {
                        'current_stage': self.env.env.current_stage,
                        'episode_count': self.env.env.episode_count,
                        'episode_count_per_stage': self.env.env.episode_count_per_stage.copy() if hasattr(self.env.env, 'episode_count_per_stage') else {},
                        'timesteps_count_per_stage': self.env.env.timesteps_count_per_stage.copy() if hasattr(self.env.env, 'timesteps_count_per_stage') else {},
                        'recent_episodes_success': self.env.env.recent_episodes_success.copy() if hasattr(self.env.env, 'recent_episodes_success') else [],
                        'recent_episodes_detection': self.env.env.recent_episodes_detection.copy() if hasattr(self.env.env, 'recent_episodes_detection') else [],
                        'recent_episodes_tracking': self.env.env.recent_episodes_tracking.copy() if hasattr(self.env.env, 'recent_episodes_tracking') else [],
                        'recent_episodes_rewards': self.env.env.recent_episodes_rewards.copy() if hasattr(self.env.env, 'recent_episodes_rewards') else []
                    }
            return None
        
        def _on_step(self) -> bool:
            # Sauvegarder l'état du curriculum à chaque checkpoint (même fréquence que CheckpointCallback)
            if self.n_calls % CHECKPOINT_SAVE_FREQ == 0:
                curriculum_state = self._get_curriculum_state()
                if curriculum_state is not None:
                    # Sauvegarder avec le même nom que le checkpoint (sans extension .zip)
                    checkpoint_name = f"ppo_marl_checkpoint_{self.num_timesteps}_steps"
                    curriculum_state_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_curriculum_state.json")
                    with open(curriculum_state_path, 'w') as f:
                        json.dump(curriculum_state, f, indent=2)
                    if self.verbose > 0:
                        print(f"💾 État du curriculum sauvegardé : {curriculum_state_path}")
            return True
    
    curriculum_callback = CurriculumStateCallback(env, checkpoint_dir, verbose=1)
    
    # Combiner les callbacks
    callback = CallbackList([tensorboard_callback, checkpoint_callback, curriculum_callback])
    
    print(f"💾 Checkpoints sauvegardés toutes les {CHECKPOINT_SAVE_FREQ:,} steps dans : {checkpoint_dir}")
    
    # Apprentissage
    total_timesteps = EPISODES * MAX_STEPS_PER_EPISODE
    print(f"🚀 Training PPO for {total_timesteps:,} timesteps...")
    print(f"📊 TensorBoard disponible dans : {session_log_dir}")
    print(f"💡 Pour visualiser : tensorboard --logdir {session_log_dir}")
    
    # Démarrer TensorBoard automatiquement (optionnel)
    # Note : Si le port est déjà utilisé, TensorBoard ne démarrera pas mais l'entraînement continuera
    try:
        start_tensorboard(TENSORBOARD_LOG_DIR, port=6006)
        print(f"💡 TensorBoard est démarré automatiquement. Vous pouvez le fermer, les métriques seront toujours sauvegardées.")
        print(f"💡 Pour relancer TensorBoard plus tard : tensorboard --logdir {session_log_dir}")
    except Exception as e:
        print(f"⚠️  Impossible de démarrer TensorBoard automatiquement : {e}")
        print(f"   → L'entraînement continuera normalement, les métriques seront sauvegardées dans les fichiers CSV")
        print(f"   → Vous pouvez démarrer TensorBoard manuellement avec : tensorboard --logdir {session_log_dir}")
    
    # Désactiver la barre de progression si tqdm/rich ne sont pas installés
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        print("⚠️  tqdm/rich non installés. Barre de progression désactivée.")
        print("   Pour l'activer : pip install tqdm rich")
        use_progress_bar = False
    
    # Sauvegarde d'urgence en cas d'interruption
    import signal
    import atexit
    
    def save_emergency_model():
        """Sauvegarde d'urgence du modèle en cas de crash ou interruption."""
        try:
            emergency_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_emergency_{timestamp}")
            model.save(emergency_path)
            print(f"\n💾 Modèle d'urgence sauvegardé : {emergency_path}.zip")
        except Exception as e:
            print(f"⚠️  Impossible de sauvegarder le modèle d'urgence : {e}")
    
    # Enregistrer la fonction de sauvegarde d'urgence
    atexit.register(save_emergency_model)
    
    def signal_handler(signum, frame):
        """Gère les signaux d'interruption (Ctrl+C, etc.)."""
        print(f"\n⚠️  Interruption détectée (signal {signum})")
        save_emergency_model()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🔍 [DIAGNOSTIC] Début de model.learn()...")
    sys.stdout.flush()
    
    try:
        print("🔍 [DIAGNOSTIC] Appel de model.learn() avec les paramètres suivants :")
        print(f"   → total_timesteps: {total_timesteps:,}")
        print(f"   → callback: {type(callback).__name__}")
        print(f"   → progress_bar: {use_progress_bar}")
        sys.stdout.flush()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,  # ✅ Callback pour métriques personnalisées + checkpoints
            progress_bar=use_progress_bar
        )
        
        print("🔍 [DIAGNOSTIC] model.learn() terminé avec succès")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print(f"\n⚠️  Entraînement interrompu par l'utilisateur")
        save_emergency_model()
        raise
    except (ConnectionError, TimeoutError, OSError) as e:
        print(f"\n❌ Erreur de connexion Unity : {e}")
        print(f"   → Vérifiez que les builds Unity sont toujours actifs")
        print(f"   → Utilisez : .\\check_unity_builds.ps1")
        save_emergency_model()
        raise
    except FileNotFoundError as e:
        print(f"\n❌ Erreur de fichier (TensorBoard) : {e}")
        print(f"   → Tentative de correction...")
        try:
            # Réinitialiser le logger TensorBoard
            if hasattr(model, 'logger'):
                if hasattr(model.logger, 'close'):
                    model.logger.close()
                from stable_baselines3.common.logger import configure
                model.logger = configure(session_log_dir, ["stdout", "csv", "tensorboard"])
            print(f"   → Logger TensorBoard réinitialisé")
            print(f"   → Relancez l'entraînement")
        except Exception as fix_error:
            print(f"   → Impossible de corriger : {fix_error}")
        save_emergency_model()
        raise
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement : {e}")
        import traceback
        print(f"\n📋 Traceback complet :")
        traceback.print_exc()
        save_emergency_model()
        raise

    # ======================================================================
    # 💾 4. SAUVEGARDE DU MODÈLE
    # ======================================================================
    # Sauvegarde du modèle avec timestamp pour éviter d'écraser les précédents
    model_path_timestamped = os.path.join(SAVE_DIR, f"{MODEL_NAME}_{timestamp}")
    model.save(model_path_timestamped)
    print(f"✅ Model saved at {model_path_timestamped}.zip")
    
    # Sauvegarder l'état du curriculum avec le modèle
    try:
        if hasattr(env, 'env') and hasattr(env.env, 'get_curriculum_state'):
            curriculum_state = env.env.get_curriculum_state()
            if curriculum_state:
                curriculum_state_path = f"{model_path_timestamped}_curriculum_state.json"
                with open(curriculum_state_path, 'w', encoding='utf-8') as f:
                    json.dump(curriculum_state, f, indent=2, ensure_ascii=False)
                print(f"✅ État du curriculum sauvegardé : {curriculum_state_path}")
        elif hasattr(env, 'get_attr'):  # VecEnv
            try:
                curriculum_state = curriculum_callback._get_curriculum_state()
                if curriculum_state:
                    curriculum_state_path = f"{model_path_timestamped}_curriculum_state.json"
                    with open(curriculum_state_path, 'w', encoding='utf-8') as f:
                        json.dump(curriculum_state, f, indent=2, ensure_ascii=False)
                    print(f"✅ État du curriculum sauvegardé : {curriculum_state_path}")
            except:
                pass
    except Exception as e:
        print(f"⚠️  Impossible de sauvegarder l'état du curriculum : {e}")
    
    # Sauvegarde aussi sous le nom standard (dernier modèle)
    model_path_standard = os.path.join(SAVE_DIR, MODEL_NAME)
    model.save(model_path_standard)
    print(f"✅ Model also saved as {model_path_standard}.zip (latest model)")
    
    # Sauvegarde du modèle dans le dossier d'expérimentation
    experiment_model_path = os.path.join(experiment_dir, f"{MODEL_NAME}_{timestamp}.zip")
    if os.path.exists(f"{model_path_timestamped}.zip"):
        shutil.copy2(f"{model_path_timestamped}.zip", experiment_model_path)
        print(f"✅ Model also saved in experiment dir : {experiment_model_path}")
    else:
        print(f"⚠️  Modèle non trouvé pour copie dans experiment dir : {model_path_timestamped}.zip")
    
    # ======================================================================
    # 📊 5. SAUVEGARDE DES INFORMATIONS D'EXPÉRIMENTATION
    # ======================================================================
    model_info = {
        "model_name": MODEL_NAME,
        "model_path_timestamped": f"{model_path_timestamped}.zip",
        "model_path_standard": f"{model_path_standard}.zip",
        "model_path_experiment": experiment_model_path,
        "total_timesteps": EPISODES * MAX_STEPS_PER_EPISODE,
        "episodes": EPISODES,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "device": device,
        "num_drones": NUM_DRONES,
    }
    
    save_experiment_info(experiment_dir, config_dict, model_info, timestamp)
    
    # ======================================================================
    # 📊 6. GÉNÉRATION DU RAPPORT D'ANALYSE
    # ======================================================================
    generate_analysis_report(experiment_dir, timestamp, env)
    
    # ======================================================================
    # 🎯 7. RÉSUMÉ DE L'EXPÉRIMENTATION
    # ======================================================================
    print("\n" + "=" * 70)
    print("✅ EXPÉRIMENTATION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(f"📁 Dossier d'expérimentation : {experiment_dir}")
    print(f"🤖 Modèle sauvegardé : {model_path_timestamped}.zip")
    print(f"📊 Logs TensorBoard : {session_log_dir}")
    print(f"📝 Configuration : {config_path}")
    print(f"📦 Fichiers archivés : {len(archived_files)} fichiers")
    print("=" * 70)
    print(f"\n💡 Pour visualiser TensorBoard : tensorboard --logdir {session_log_dir}")
    print(f"💡 Pour évaluer le modèle : python evaluate_marl.py --model {MODEL_NAME}_{timestamp}.zip")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement PPO pour patrouille multi-drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Nouvel entraînement
  python train_marl.py
  
  # Continuer un entraînement existant (recommandé)
  python train_marl.py --continue_training models/ppo_marl_20251204_115027.zip
  
  # OU utiliser --load_model (même effet que --continue_training)
  python train_marl.py --load_model models/ppo_marl_20251204_115027.zip
  
  # Repartir de zéro avec un modèle existant (curriculum réinitialisé)
  python train_marl.py --load_model models/ppo_marl_20251204_115027.zip --reset_curriculum
        """
    )
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='Chemin vers un modèle à charger (ex: models/ppo_marl_20251204_115027.zip). Restaure automatiquement l\'état du curriculum si disponible.'
    )
    parser.add_argument(
        '--continue_training',
        type=str,
        default=None,
        metavar='MODEL_PATH',
        help='Continuer un entraînement existant (alias de --load_model). Restaure automatiquement l\'état du curriculum.'
    )
    parser.add_argument(
        '--reset_curriculum',
        action='store_true',
        help='Réinitialiser le curriculum à zéro (utilisé avec --load_model ou --continue_training)'
    )
    
    args = parser.parse_args()
    
    # --continue_training est un alias de --load_model
    load_model_path = args.continue_training if args.continue_training else args.load_model
    
    train(load_model_path=load_model_path, reset_curriculum=args.reset_curriculum)
