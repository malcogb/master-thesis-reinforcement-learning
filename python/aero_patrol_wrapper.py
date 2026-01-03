import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Imports avec gestion flexible du chemin
# Essayer d'abord les imports relatifs (si dans le package python)
# Sinon utiliser les imports absolus (si lancé depuis marl/ ou racine)
try:
    # Essayer d'abord les imports relatifs (fonctionne si importé comme package)
    from .env_manager import UnityEnvManager
    from .config import (
        DETECTION_RADIUS, GRID_SIZE, COVERAGE_RADIUS,
        ZONE_MIN_X, ZONE_MAX_X, ZONE_MIN_Z, ZONE_MAX_Z,
        REWARD_COVERAGE, REWARD_DETECTION, REWARD_OVERLAP_PENALTY,
        REWARD_COLLISION_PENALTY, REWARD_INTRUDER_COLLISION_PENALTY, REWARD_DISTANCE_PENALTY,
        TRACKING_RADIUS, TRACKING_REWARD, CAPTURE_RADIUS, CAPTURE_DRONES_REQUIRED,
        REWARD_CAPTURE, ZONE_BOUNDARY_MARGIN, REWARD_OUT_OF_ZONE_PENALTY,
        REWARD_BOUNDARY_WARNING, REWARD_CENTRAL_ALERT, CENTRAL_ALERT_COOLDOWN,
        REWARD_IN_ZONE, ZONE_TRUNCATION_MARGIN,
        MIN_DRONE_SEPARATION, OPTIMAL_DRONE_SEPARATION, MAX_DRONE_SEPARATION,
        REWARD_GOOD_SEPARATION, REWARD_TOO_CLOSE_PENALTY, REWARD_TOO_FAR_PENALTY,
        OBSTACLE_COLLISION_RADIUS, REWARD_OBSTACLE_COLLISION_PENALTY,
        REWARD_OBSTACLE_NEAR_PENALTY, OBSTACLE_NEAR_THRESHOLD,
        # Curriculum learning
        CURRICULUM_ENABLED, CURRICULUM_START_STAGE,
        CURRICULUM_STAGE0_INTRUDER_SPEED_MULT, CURRICULUM_STAGE0_RADIUS_MULT,
        CURRICULUM_STAGE0_ENABLE_COVERAGE, CURRICULUM_STAGE0_ENABLE_OBSTACLES,
        CURRICULUM_STAGE0_ENABLE_ZONE, CURRICULUM_STAGE0_ENABLE_SEPARATION,
        CURRICULUM_STAGE0_ENABLE_CENTRAL_ALERT, CURRICULUM_STAGE0_ENABLE_DISTANCE_PENALTY,
        CURRICULUM_STAGE1_INTRUDER_SPEED_MULT, CURRICULUM_STAGE1_RADIUS_MULT,
        CURRICULUM_STAGE1_ENABLE_COVERAGE, CURRICULUM_STAGE1_ENABLE_OBSTACLES,
        CURRICULUM_STAGE1_ENABLE_ZONE, CURRICULUM_STAGE1_ENABLE_SEPARATION,
        CURRICULUM_STAGE1_ENABLE_CENTRAL_ALERT, CURRICULUM_STAGE1_ENABLE_DISTANCE_PENALTY,
        CURRICULUM_STAGE2_INTRUDER_SPEED_MULT, CURRICULUM_STAGE2_RADIUS_MULT,
        CURRICULUM_STAGE2_ENABLE_COVERAGE, CURRICULUM_STAGE2_ENABLE_OBSTACLES,
        CURRICULUM_STAGE2_ENABLE_ZONE, CURRICULUM_STAGE2_ENABLE_SEPARATION,
        CURRICULUM_STAGE2_ENABLE_CENTRAL_ALERT, CURRICULUM_STAGE2_ENABLE_DISTANCE_PENALTY,
        CURRICULUM_STAGE0_SUCCESS_RATE_THRESHOLD, CURRICULUM_STAGE0_MIN_EPISODES,
        CURRICULUM_STAGE0_MIN_TIMESTEPS,
        CURRICULUM_STAGE0_DETECTION_RATE_THRESHOLD, CURRICULUM_STAGE0_TRACKING_RATE_THRESHOLD,
        CURRICULUM_STAGE0_STABILITY_THRESHOLD,
        CURRICULUM_STAGE1_SUCCESS_RATE_THRESHOLD, CURRICULUM_STAGE1_MIN_EPISODES,
        CURRICULUM_STAGE1_MIN_TIMESTEPS,
        CURRICULUM_STAGE1_DETECTION_RATE_THRESHOLD, CURRICULUM_STAGE1_TRACKING_RATE_THRESHOLD,
        CURRICULUM_STAGE1_STABILITY_THRESHOLD,
        CURRICULUM_EVALUATION_WINDOW,
        REWARD_SCALING_ENABLED, REWARD_SCALING_FACTOR
    )
except (ImportError, ValueError):
    # Fallback : imports absolus (fonctionne si le répertoire parent est dans sys.path)
    # Cela fonctionne car train_marl.py ajoute le répertoire parent à sys.path
    import sys
    import os
    
    # S'assurer que le répertoire parent est dans sys.path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_file_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from python.env_manager import UnityEnvManager
    from python.config import (
        DETECTION_RADIUS, GRID_SIZE, COVERAGE_RADIUS,
        ZONE_MIN_X, ZONE_MAX_X, ZONE_MIN_Z, ZONE_MAX_Z,
        REWARD_COVERAGE, REWARD_DETECTION, REWARD_OVERLAP_PENALTY,
        REWARD_COLLISION_PENALTY, REWARD_INTRUDER_COLLISION_PENALTY, REWARD_DISTANCE_PENALTY,
        TRACKING_RADIUS, TRACKING_REWARD, CAPTURE_RADIUS, CAPTURE_DRONES_REQUIRED,
        REWARD_CAPTURE, ZONE_BOUNDARY_MARGIN, REWARD_OUT_OF_ZONE_PENALTY,
        REWARD_BOUNDARY_WARNING, REWARD_CENTRAL_ALERT, CENTRAL_ALERT_COOLDOWN,
        REWARD_IN_ZONE, ZONE_TRUNCATION_MARGIN,
        MIN_DRONE_SEPARATION, OPTIMAL_DRONE_SEPARATION, MAX_DRONE_SEPARATION,
        REWARD_GOOD_SEPARATION, REWARD_TOO_CLOSE_PENALTY, REWARD_TOO_FAR_PENALTY,
        OBSTACLE_COLLISION_RADIUS, REWARD_OBSTACLE_COLLISION_PENALTY,
        REWARD_OBSTACLE_NEAR_PENALTY, OBSTACLE_NEAR_THRESHOLD,
        # Curriculum learning
        CURRICULUM_ENABLED, CURRICULUM_START_STAGE,
        CURRICULUM_STAGE0_INTRUDER_SPEED_MULT, CURRICULUM_STAGE0_RADIUS_MULT,
        CURRICULUM_STAGE0_ENABLE_COVERAGE, CURRICULUM_STAGE0_ENABLE_OBSTACLES,
        CURRICULUM_STAGE0_ENABLE_ZONE, CURRICULUM_STAGE0_ENABLE_SEPARATION,
        CURRICULUM_STAGE0_ENABLE_CENTRAL_ALERT, CURRICULUM_STAGE0_ENABLE_DISTANCE_PENALTY,
        CURRICULUM_STAGE1_INTRUDER_SPEED_MULT, CURRICULUM_STAGE1_RADIUS_MULT,
        CURRICULUM_STAGE1_ENABLE_COVERAGE, CURRICULUM_STAGE1_ENABLE_OBSTACLES,
        CURRICULUM_STAGE1_ENABLE_ZONE, CURRICULUM_STAGE1_ENABLE_SEPARATION,
        CURRICULUM_STAGE1_ENABLE_CENTRAL_ALERT, CURRICULUM_STAGE1_ENABLE_DISTANCE_PENALTY,
        CURRICULUM_STAGE2_INTRUDER_SPEED_MULT, CURRICULUM_STAGE2_RADIUS_MULT,
        CURRICULUM_STAGE2_ENABLE_COVERAGE, CURRICULUM_STAGE2_ENABLE_OBSTACLES,
        CURRICULUM_STAGE2_ENABLE_ZONE, CURRICULUM_STAGE2_ENABLE_SEPARATION,
        CURRICULUM_STAGE2_ENABLE_CENTRAL_ALERT, CURRICULUM_STAGE2_ENABLE_DISTANCE_PENALTY,
        CURRICULUM_STAGE0_SUCCESS_RATE_THRESHOLD, CURRICULUM_STAGE0_MIN_EPISODES,
        CURRICULUM_STAGE0_MIN_TIMESTEPS,
        CURRICULUM_STAGE0_DETECTION_RATE_THRESHOLD, CURRICULUM_STAGE0_TRACKING_RATE_THRESHOLD,
        CURRICULUM_STAGE0_STABILITY_THRESHOLD,
        CURRICULUM_STAGE1_SUCCESS_RATE_THRESHOLD, CURRICULUM_STAGE1_MIN_EPISODES,
        CURRICULUM_STAGE1_MIN_TIMESTEPS,
        CURRICULUM_STAGE1_DETECTION_RATE_THRESHOLD, CURRICULUM_STAGE1_TRACKING_RATE_THRESHOLD,
        CURRICULUM_STAGE1_STABILITY_THRESHOLD,
        CURRICULUM_EVALUATION_WINDOW,
        REWARD_SCALING_ENABLED, REWARD_SCALING_FACTOR
    )


class AeroPatrolWrapper(gym.Env):
    """
    Environnement de patrouille multi-drone pour détection d'intrusion.
    
    Scénario MARL :
    - Coordination décentralisée entre drones
    - Couverture optimale de la zone (minimiser zones non observées)
    - Détection et signalement d'intrusion
    - Observation totale (tous les agents voient l'état global)
    
    Note: L'observation partielle sera implémentée ultérieurement pour
    ajouter le défi de la coordination avec information limitée.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, num_drones=3, max_steps=500, port=None):
        super().__init__()
        # Utiliser le port fourni ou le port par défaut depuis config
        if port is None:
            from python.config import UNITY_PORT
            port = UNITY_PORT
        print(f"🔍 [DIAGNOSTIC] AeroPatrolWrapper.__init__ : Création de UnityEnvManager sur le port {port}...")
        import sys
        sys.stdout.flush()
        try:
            self.manager = UnityEnvManager(port=port)
            print(f"🔍 [DIAGNOSTIC] AeroPatrolWrapper.__init__ : UnityEnvManager créé avec succès")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ [DIAGNOSTIC] AeroPatrolWrapper.__init__ : Erreur lors de la création de UnityEnvManager : {e}")
            sys.stdout.flush()
            raise
        self.num_drones = num_drones
        self.max_steps = max_steps
        self.current_step = 0
        
        # Curriculum Learning
        self.curriculum_enabled = CURRICULUM_ENABLED
        self.current_stage = CURRICULUM_START_STAGE if self.curriculum_enabled else 2  # Stage 2 = tout activé si curriculum désactivé
        self.episode_count = 0  # Compteur global d'épisodes
        self.episode_count_per_stage = {0: 0, 1: 0, 2: 0}  # Compteur d'épisodes par stage
        self.timesteps_count_per_stage = {0: 0, 1: 0, 2: 0}  # Compteur de timesteps par stage
        self.recent_episodes_success = []  # Historique des succès (capture) pour progression
        self.recent_episodes_detection = []  # Historique des détections pour progression
        self.recent_episodes_tracking = []  # Historique des trackings pour progression
        self.recent_episodes_rewards = []  # Historique des récompenses totales d'épisode pour calcul de stabilité
        self.episode_total_reward = 0.0  # Accumulateur de récompense pour l'épisode actuel
        
        # Grille de couverture pour tracker les zones surveillées
        self.grid_size = GRID_SIZE
        self.cell_size_x = (ZONE_MAX_X - ZONE_MIN_X) / GRID_SIZE
        self.cell_size_z = (ZONE_MAX_Z - ZONE_MIN_Z) / GRID_SIZE
        self.coverage_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.coverage_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)  # Nombre de drones couvrant chaque cellule
        
        # État de détection et tracking
        self.intruder_detected = False
        self.detection_step = -1
        self.intruder_captured = False
        self.capture_step = -1
        self.central_alerted = False
        self.last_central_alert_step = -CENTRAL_ALERT_COOLDOWN
        self.tracking_drones = []  # Liste des drones qui suivent l'intrus
        self.current_obstacles = []  # Liste actuelle des obstacles (positions + radius)
        self.previous_drone_positions = None  # Positions précédentes des drones pour calcul de vitesse (2025-12-07)
        
        # Espaces d'action et d'observation
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_drones, 2), dtype=np.float32
        )
        
        # Observation enrichie : positions drones + intruder + obstacles + état détection + couverture locale
        # Note: Dimension dynamique pour obstacles (max 10 obstacles par défaut, peut être ajusté)
        self.max_obstacles = 10  # Nombre maximum d'obstacles supportés
        obs_dim = (num_drones * 3 + 3 + 1 + num_drones + 
                   self.max_obstacles * 4)  # drones (x,y,z) + intruder (x,y,z) + detected + couverture_locale + obstacles (x,y,z,radius)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_stage_config(self):
        """Retourne la configuration du stage actuel."""
        if not self.curriculum_enabled:
            # Si curriculum désactivé, utiliser la config du stage 2 (tout activé)
            return {
                'intruder_speed_mult': 1.0,
                'radius_mult': 1.0,
                'enable_coverage': True,
                'enable_obstacles': True,
                'enable_zone': True,
                'enable_separation': True,
                'enable_central_alert': True,
                'enable_distance_penalty': True
            }
        
        if self.current_stage == 0:
            return {
                'intruder_speed_mult': CURRICULUM_STAGE0_INTRUDER_SPEED_MULT,
                'radius_mult': CURRICULUM_STAGE0_RADIUS_MULT,
                'enable_coverage': CURRICULUM_STAGE0_ENABLE_COVERAGE,
                'enable_obstacles': CURRICULUM_STAGE0_ENABLE_OBSTACLES,
                'enable_zone': CURRICULUM_STAGE0_ENABLE_ZONE,
                'enable_separation': CURRICULUM_STAGE0_ENABLE_SEPARATION,
                'enable_central_alert': CURRICULUM_STAGE0_ENABLE_CENTRAL_ALERT,
                'enable_distance_penalty': CURRICULUM_STAGE0_ENABLE_DISTANCE_PENALTY
            }
        elif self.current_stage == 1:
            return {
                'intruder_speed_mult': CURRICULUM_STAGE1_INTRUDER_SPEED_MULT,
                'radius_mult': CURRICULUM_STAGE1_RADIUS_MULT,
                'enable_coverage': CURRICULUM_STAGE1_ENABLE_COVERAGE,
                'enable_obstacles': CURRICULUM_STAGE1_ENABLE_OBSTACLES,
                'enable_zone': CURRICULUM_STAGE1_ENABLE_ZONE,
                'enable_separation': CURRICULUM_STAGE1_ENABLE_SEPARATION,
                'enable_central_alert': CURRICULUM_STAGE1_ENABLE_CENTRAL_ALERT,
                'enable_distance_penalty': CURRICULUM_STAGE1_ENABLE_DISTANCE_PENALTY
            }
        else:  # stage 2
            return {
                'intruder_speed_mult': CURRICULUM_STAGE2_INTRUDER_SPEED_MULT,
                'radius_mult': CURRICULUM_STAGE2_RADIUS_MULT,
                'enable_coverage': CURRICULUM_STAGE2_ENABLE_COVERAGE,
                'enable_obstacles': CURRICULUM_STAGE2_ENABLE_OBSTACLES,
                'enable_zone': CURRICULUM_STAGE2_ENABLE_ZONE,
                'enable_separation': CURRICULUM_STAGE2_ENABLE_SEPARATION,
                'enable_central_alert': CURRICULUM_STAGE2_ENABLE_CENTRAL_ALERT,
                'enable_distance_penalty': CURRICULUM_STAGE2_ENABLE_DISTANCE_PENALTY
            }
    
    def _check_stage_progression(self, episode_success, episode_detection, episode_tracking, episode_reward):
        """
        Vérifie si les critères de progression sont atteints et met à jour le stage.
        Prend en compte : success_rate (capture), detection_rate, tracking_rate, et stabilité.
        """
        if not self.curriculum_enabled:
            return
        
        # Ajouter les résultats de cet épisode
        self.recent_episodes_success.append(episode_success)  # Capture (0 ou 1)
        self.recent_episodes_detection.append(episode_detection)  # Détection (0 ou 1)
        self.recent_episodes_tracking.append(episode_tracking)  # Tracking (0 ou 1)
        self.recent_episodes_rewards.append(episode_reward)  # Récompense totale pour stabilité
        
        # Garder seulement les N derniers épisodes
        if len(self.recent_episodes_success) > CURRICULUM_EVALUATION_WINDOW:
            self.recent_episodes_success.pop(0)
            self.recent_episodes_detection.pop(0)
            self.recent_episodes_tracking.pop(0)
            self.recent_episodes_rewards.pop(0)
        
        # Vérifier la progression selon le stage actuel
        # IMPORTANT: La progression nécessite success_rate >= 70% + detection_rate >= 70% + 
        # tracking_rate >= 70% + stabilité (variance faible) + minimum 50000 timesteps
        if self.current_stage == 0:
            # Passer au stage 1 si tous les critères sont atteints
            stage_episodes = self.episode_count_per_stage.get(0, 0)
            stage_timesteps = self.timesteps_count_per_stage.get(0, 0)
            if (stage_episodes >= CURRICULUM_STAGE0_MIN_EPISODES and 
                stage_timesteps >= CURRICULUM_STAGE0_MIN_TIMESTEPS and
                len(self.recent_episodes_success) >= CURRICULUM_EVALUATION_WINDOW):
                
                # Calculer les métriques
                success_rate = np.mean(self.recent_episodes_success)  # Capture rate
                detection_rate = np.mean(self.recent_episodes_detection)  # Detection rate
                tracking_rate = np.mean(self.recent_episodes_tracking)  # Tracking rate
                
                # Calculer la stabilité (coefficient de variation des récompenses)
                rewards_array = np.array(self.recent_episodes_rewards)
                if len(rewards_array) > 0 and np.mean(rewards_array) != 0:
                    stability_cv = np.std(rewards_array) / np.abs(np.mean(rewards_array))  # Coefficient de variation
                else:
                    stability_cv = float('inf')  # Instable si pas de récompenses
                
                # Vérifier tous les critères
                success_ok = success_rate >= CURRICULUM_STAGE0_SUCCESS_RATE_THRESHOLD
                detection_ok = detection_rate >= CURRICULUM_STAGE0_DETECTION_RATE_THRESHOLD
                tracking_ok = tracking_rate >= CURRICULUM_STAGE0_TRACKING_RATE_THRESHOLD
                stability_ok = stability_cv <= CURRICULUM_STAGE0_STABILITY_THRESHOLD
                
                if success_ok and detection_ok and tracking_ok and stability_ok:
                    self.current_stage = 1
                    print(f"🎓 Curriculum: Progression au Stage 1")
                    print(f"   ✅ Capture: {success_rate:.2%} >= {CURRICULUM_STAGE0_SUCCESS_RATE_THRESHOLD:.0%}")
                    print(f"   ✅ Détection: {detection_rate:.2%} >= {CURRICULUM_STAGE0_DETECTION_RATE_THRESHOLD:.0%}")
                    print(f"   ✅ Tracking: {tracking_rate:.2%} >= {CURRICULUM_STAGE0_TRACKING_RATE_THRESHOLD:.0%}")
                    print(f"   ✅ Stabilité: CV={stability_cv:.3f} <= {CURRICULUM_STAGE0_STABILITY_THRESHOLD:.0%}")
                    print(f"   ✅ Timesteps: {stage_timesteps} >= {CURRICULUM_STAGE0_MIN_TIMESTEPS}")
                    print(f"   → Obstacles, Plane et Zone seront activés au prochain reset()")
                    print(f"   → Le modèle continue l'apprentissage sans réinitialisation (continuité préservée)")
                    # Réinitialiser pour le nouveau stage
                    self.recent_episodes_success = []
                    self.recent_episodes_detection = []
                    self.recent_episodes_tracking = []
                    self.recent_episodes_rewards = []
        
        elif self.current_stage == 1:
            # Passer au stage 2 si tous les critères sont atteints
            stage_episodes = self.episode_count_per_stage.get(1, 0)
            stage_timesteps = self.timesteps_count_per_stage.get(1, 0)
            if (stage_episodes >= CURRICULUM_STAGE1_MIN_EPISODES and 
                stage_timesteps >= CURRICULUM_STAGE1_MIN_TIMESTEPS and
                len(self.recent_episodes_success) >= CURRICULUM_EVALUATION_WINDOW):
                
                # Calculer les métriques
                success_rate = np.mean(self.recent_episodes_success)  # Capture rate
                detection_rate = np.mean(self.recent_episodes_detection)  # Detection rate
                tracking_rate = np.mean(self.recent_episodes_tracking)  # Tracking rate
                
                # Calculer la stabilité (coefficient de variation des récompenses)
                rewards_array = np.array(self.recent_episodes_rewards)
                if len(rewards_array) > 0 and np.mean(rewards_array) != 0:
                    stability_cv = np.std(rewards_array) / np.abs(np.mean(rewards_array))  # Coefficient de variation
                else:
                    stability_cv = float('inf')  # Instable si pas de récompenses
                
                # Vérifier tous les critères
                success_ok = success_rate >= CURRICULUM_STAGE1_SUCCESS_RATE_THRESHOLD
                detection_ok = detection_rate >= CURRICULUM_STAGE1_DETECTION_RATE_THRESHOLD
                tracking_ok = tracking_rate >= CURRICULUM_STAGE1_TRACKING_RATE_THRESHOLD
                stability_ok = stability_cv <= CURRICULUM_STAGE1_STABILITY_THRESHOLD
                
                if success_ok and detection_ok and tracking_ok and stability_ok:
                    self.current_stage = 2
                    print(f"🎓 Curriculum: Progression au Stage 2")
                    print(f"   ✅ Capture: {success_rate:.2%} >= {CURRICULUM_STAGE1_SUCCESS_RATE_THRESHOLD:.0%}")
                    print(f"   ✅ Détection: {detection_rate:.2%} >= {CURRICULUM_STAGE1_DETECTION_RATE_THRESHOLD:.0%}")
                    print(f"   ✅ Tracking: {tracking_rate:.2%} >= {CURRICULUM_STAGE1_TRACKING_RATE_THRESHOLD:.0%}")
                    print(f"   ✅ Stabilité: CV={stability_cv:.3f} <= {CURRICULUM_STAGE1_STABILITY_THRESHOLD:.0%}")
                    print(f"   ✅ Timesteps: {stage_timesteps} >= {CURRICULUM_STAGE1_MIN_TIMESTEPS}")
                    print(f"   → Coverage et Séparation seront activés au prochain reset()")
                    print(f"   → Le modèle continue l'apprentissage sans réinitialisation (continuité préservée)")
                    # Réinitialiser pour le nouveau stage
                    self.recent_episodes_success = []
                    self.recent_episodes_detection = []
                    self.recent_episodes_tracking = []
                    self.recent_episodes_rewards = []

    def reset_curriculum_state(self, stage=None, episode_count=None, episode_count_per_stage=None, 
                               timesteps_count_per_stage=None, recent_episodes_success=None,
                               recent_episodes_detection=None, recent_episodes_tracking=None,
                               recent_episodes_rewards=None):
        """Réinitialise l'état du curriculum (méthode utilisable avec SubprocVecEnv.set_attr)."""
        if stage is not None:
            self.current_stage = stage
        if episode_count is not None:
            self.episode_count = episode_count
        if episode_count_per_stage is not None:
            self.episode_count_per_stage = episode_count_per_stage
        if timesteps_count_per_stage is not None:
            self.timesteps_count_per_stage = timesteps_count_per_stage
        if recent_episodes_success is not None:
            self.recent_episodes_success = recent_episodes_success
        if recent_episodes_detection is not None:
            self.recent_episodes_detection = recent_episodes_detection
        if recent_episodes_tracking is not None:
            self.recent_episodes_tracking = recent_episodes_tracking
        if recent_episodes_rewards is not None:
            self.recent_episodes_rewards = recent_episodes_rewards
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement pour un nouvel épisode."""
        import sys
        print(f"🔍 [DIAGNOSTIC] AeroPatrolWrapper.reset : Début...")
        sys.stdout.flush()
        
        super().reset(seed=seed)
        
        # Obtenir la configuration du stage actuel
        stage_config = self._get_stage_config()
        
        # 🎓 Envoyer la configuration du stage à Unity (pour vitesse intrus et gestion obstacles/zone)
        # Note: Les drones sont créés UNIQUEMENT au démarrage Unity, pas de mise à jour pendant l'entraînement
        # IMPORTANT: Les obstacles, plane et zone ne sont activés que lorsque TOUS les critères sont atteints
        # (vérifié dans _check_stage_progression à la fin de chaque épisode).
        # - Stage 0 → Stage 1: obstacles/zone activés seulement si:
        #   * Capture rate >= 70% + Detection rate >= 70% + Tracking rate >= 70% + Stabilité (CV <= 15%) + 50000 timesteps minimum
        # - Stage 1 → Stage 2: coverage/séparation activés seulement si:
        #   * Capture rate >= 70% + Detection rate >= 70% + Tracking rate >= 70% + Stabilité (CV <= 15%) + 50000 timesteps minimum
        # NOTE: Le modèle PPO continue l'apprentissage sans réinitialisation entre les stages (continuité préservée)
        print(f"🔍 [DIAGNOSTIC] AeroPatrolWrapper.reset : Appel de manager.reset()...")
        sys.stdout.flush()
        try:
            obs_json = self.manager.reset(
                stage=self.current_stage,
                intruder_speed_mult=stage_config['intruder_speed_mult'],
                enable_obstacles=stage_config['enable_obstacles'],
                enable_zone=stage_config['enable_zone']
            )
            print(f"🔍 [DIAGNOSTIC] AeroPatrolWrapper.reset : manager.reset() terminé")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ [DIAGNOSTIC] AeroPatrolWrapper.reset : Erreur lors de manager.reset() : {e}")
            sys.stdout.flush()
            raise
        obs = self._process_obs(obs_json)
        self.current_step = 0
        
        # Réinitialiser la grille de couverture
        self.coverage_grid.fill(0.0)
        self.coverage_count.fill(0)
        self.intruder_detected = False
        self.detection_step = -1
        self.intruder_captured = False
        
        # Initialiser les positions précédentes pour calcul de vitesse (2025-12-07)
        drones = np.array([[d["x"], d["y"], d["z"]] for d in obs_json["drones"]])
        self.previous_drone_positions = drones.copy()
        self.capture_step = -1
        self.central_alerted = False
        self.last_central_alert_step = -CENTRAL_ALERT_COOLDOWN
        self.tracking_drones = []
        self.current_obstacles = []
        self.episode_total_reward = 0.0  # Réinitialiser l'accumulateur de récompense
        
        return obs, {}

    def step(self, action):
        """Exécute une étape de l'environnement."""
        import sys
        self.current_step += 1
        
        # Log de diagnostic seulement toutes les 100 steps pour ne pas surcharger
        if self.current_step % 100 == 0:
            print(f"🔍 [DIAGNOSTIC] AeroPatrolWrapper.step : Step {self.current_step}...")
            sys.stdout.flush()
        
        try:
            obs_json = self.manager.step(action)
            if self.current_step % 100 == 0:
                print(f"🔍 [DIAGNOSTIC] AeroPatrolWrapper.step : Step {self.current_step} terminé")
                sys.stdout.flush()
        except Exception as e:
            print(f"❌ [DIAGNOSTIC] AeroPatrolWrapper.step : Erreur à la step {self.current_step} : {e}")
            sys.stdout.flush()
            raise
        obs = self._process_obs(obs_json)
        
        # Extraire les positions
        intruder = np.array([obs_json["intruder"]["x"], 
                            obs_json["intruder"]["y"], 
                            obs_json["intruder"]["z"]])
        drones = np.array([[d["x"], d["y"], d["z"]] for d in obs_json["drones"]])
        
        # Réinitialiser le comptage de couverture pour ce step
        self.coverage_count.fill(0)
        
        # Calculer les récompenses selon le scénario de patrouille
        reward, info = self._compute_patrol_rewards(drones, intruder)
        
        # Accumuler la récompense totale de l'épisode pour le calcul de stabilité
        self.episode_total_reward += reward
        
        # Incrémenter le compteur de timesteps pour le stage actuel
        # S'assurer que la clé existe (sécurité si restauration incomplète)
        if self.current_stage not in self.timesteps_count_per_stage:
            self.timesteps_count_per_stage[self.current_stage] = 0
        self.timesteps_count_per_stage[self.current_stage] += 1
        
        # Vérifier les conditions de terminaison
        terminated, truncated = self._check_done(drones, intruder)
        
        # Calculer le ratio de couverture (pour compatibilité avec l'ancien code)
        coverage_ratio_from_grid = np.sum(self.coverage_grid > 0) / (self.grid_size ** 2) if self.grid_size > 0 else 0.0
        
        # Calculer les métriques de progression pour le stage actuel
        stage_success_rate = 0.0
        stage_detection_rate = 0.0
        stage_tracking_rate = 0.0
        stage_stability_cv = 0.0
        
        if len(self.recent_episodes_success) > 0:
            stage_success_rate = np.mean(self.recent_episodes_success)
            stage_detection_rate = np.mean(self.recent_episodes_detection)
            stage_tracking_rate = np.mean(self.recent_episodes_tracking)
            
            # Calculer la stabilité
            rewards_array = np.array(self.recent_episodes_rewards)
            if len(rewards_array) > 0 and np.mean(rewards_array) != 0:
                stage_stability_cv = np.std(rewards_array) / np.abs(np.mean(rewards_array))
        
        # Obtenir la configuration du stage pour TensorBoard
        stage_config = self._get_stage_config()
        
        # Ajouter les informations de curriculum
        info.update({
            "mean_distance": np.mean(np.linalg.norm(drones - intruder, axis=1)),
            "coverage_ratio": coverage_ratio_from_grid,  # Utiliser la valeur calculée
            "intruder_detected": self.intruder_detected,
            "detection_step": self.detection_step if self.intruder_detected else None,
            "intruder_captured": self.intruder_captured,
            "capture_step": self.capture_step if self.intruder_captured else None,
            "central_alerted": self.central_alerted,
            "tracking_drones_count": len(self.tracking_drones),
            # Curriculum metrics
            "curriculum_stage": self.current_stage,
            "curriculum_enabled": self.curriculum_enabled,
            "curriculum_episode_count": self.episode_count,
            "curriculum_episode_count_stage": self.episode_count_per_stage.get(self.current_stage, 0),
            "curriculum_timesteps_count_stage": self.timesteps_count_per_stage.get(self.current_stage, 0),
            # Success rates pour progression
            "curriculum_success_rate": stage_success_rate,
            "curriculum_detection_rate": stage_detection_rate,
            "curriculum_tracking_rate": stage_tracking_rate,
            "curriculum_stability_cv": stage_stability_cv,
            # Stage configuration (pour voir ce qui est activé)
            "stage_enable_coverage": int(stage_config['enable_coverage']),
            "stage_enable_obstacles": int(stage_config['enable_obstacles']),
            "stage_enable_zone": int(stage_config['enable_zone']),
            "stage_enable_separation": int(stage_config['enable_separation']),
            "stage_enable_central_alert": int(stage_config['enable_central_alert']),
            "stage_enable_distance_penalty": int(stage_config['enable_distance_penalty']),
            "stage_intruder_speed_mult": stage_config['intruder_speed_mult'],
            "stage_radius_mult": stage_config['radius_mult']
        })
        
        # Vérifier la progression de stage à la fin de l'épisode
        if terminated or truncated:
            episode_success = 1 if self.intruder_captured else 0  # Capture
            episode_detection = 1 if self.intruder_detected else 0  # Détection
            episode_tracking = 1 if len(self.tracking_drones) > 0 else 0  # Tracking (au moins un drone)
            episode_reward = self.episode_total_reward  # Récompense totale accumulée de l'épisode pour stabilité
            
            self.episode_count += 1  # Compteur global
            # S'assurer que la clé existe (sécurité si restauration incomplète)
            if self.current_stage not in self.episode_count_per_stage:
                self.episode_count_per_stage[self.current_stage] = 0
            self.episode_count_per_stage[self.current_stage] += 1  # Compteur par stage
            self._check_stage_progression(episode_success, episode_detection, episode_tracking, episode_reward)
        
        return obs, reward, terminated, truncated, info

    def _get_stage_reward_multiplier(self):
        """
        Réduire les récompenses au Stage 0 pour éviter le pic initial.
        Augmentation progressive pour obtenir une allure asymptotique.
        
        Returns:
            float: Multiplicateur pour les récompenses (0.8 au Stage 0, 1.0 après)
        """
        if self.current_stage == 0:
            # Réduire les récompenses au Stage 0 pour éviter le pic initial
            # Vérifier si l'agent s'est adapté (basé sur les timesteps)
            timesteps_in_stage0 = self.timesteps_count_per_stage.get(0, 0)
            if timesteps_in_stage0 < 25000:  # Premiers 25k timesteps
                return 0.8  # Réduire de 20% au début
            else:
                return 0.9  # Augmentation progressive
        else:
            # Stages 1 et 2 : récompenses normales
            return 1.0
    
    def _get_penalty_multiplier(self, penalty_type):
        """
        Réduire les pénalités lors des transitions pour obtenir une allure asymptotique.
        
        Logique :
        - Pénalités toujours activées (collision, distance) : pas de réduction
        - Nouvelles pénalités au Stage 1 : réduction de 50% lors de l'introduction
        - Nouvelles pénalités au Stage 2 : réduction de 50% lors de l'introduction
        - Après adaptation (50k timesteps) : retour progressif à 100%
        
        Args:
            penalty_type: 'collision', 'intruder_collision', 'distance', 'zone', 
                        'boundary', 'obstacle_collision', 'obstacle_near', 
                        'separation', 'too_close', 'too_far', 'overlap'
        
        Returns:
            float: Multiplicateur pour la pénalité (0.5-1.0)
        """
        # Pénalités toujours activées : pas de réduction
        if penalty_type in ['collision', 'intruder_collision', 'distance']:
            return 1.0  # Toujours à 100%
        
        # Pénalités ajoutées au Stage 1 : réduction au Stage 1
        if penalty_type in ['zone', 'boundary', 'obstacle_collision', 'obstacle_near']:
            if self.current_stage == 1:
                # Vérifier si l'agent s'est adapté
                timesteps_in_stage1 = self.timesteps_count_per_stage.get(1, 0)
                if timesteps_in_stage1 < 50000:  # Premiers 50k timesteps
                    return 0.5  # Réduire de 50% lors de l'introduction
                else:
                    return 1.0  # Normales après adaptation
            else:  # Stage 2
                # Au Stage 2, ces pénalités sont déjà normales
                return 1.0
        
        # Pénalités ajoutées au Stage 2 : réduction au Stage 2
        if penalty_type in ['separation', 'too_close', 'too_far', 'overlap']:
            if self.current_stage == 2:
                # Vérifier si l'agent s'est adapté
                timesteps_in_stage2 = self.timesteps_count_per_stage.get(2, 0)
                if timesteps_in_stage2 < 50000:  # Premiers 50k timesteps
                    return 0.5  # Réduire de 50% lors de l'introduction
                else:
                    return 1.0  # Normales après adaptation
            else:
                return 1.0
        
        return 1.0  # Par défaut
    
    def _get_stage_bonus(self):
        """
        Bonus progressifs pour compenser les pénalités et maintenir une courbe ascendante.
        
        Logique :
        - Stage 0 : Petit bonus pour encourager l'apprentissage initial
        - Stage 1 : Bonus moyen pour compenser les nouvelles pénalités
        - Stage 2 : Bonus plus élevé pour compenser toutes les pénalités
        
        Returns:
            float: Bonus par step selon le stage
        """
        if self.current_stage == 0:
            return 0.1  # Petit bonus au Stage 0
        elif self.current_stage == 1:
            return 0.2  # Bonus moyen au Stage 1
        else:  # Stage 2
            return 0.3  # Bonus plus élevé au Stage 2

    def _compute_patrol_rewards(self, drones, intruder):
        """
        Reward complet refactorisé et pondéré pour PPO.
        Objectifs :
        - Progression par stage (curriculum)
        - Rewards modérés pour éviter les oscillations
        - Pénalités limitées pour collisions/obstacles
        - Courbe ep_rew_mean asymptotique
        """
        stage_config = self._get_stage_config()
        total_reward = 0.0
        info = {}

        # =========================
        # 0. Multiplicateur global (pour stage progression)
        # =========================
        reward_multiplier = self._get_stage_reward_multiplier()
        
        # =========================
        # 1. Patrouille active (mouvement des drones)
        # =========================
        patrol_reward = 0.0
        if self.previous_drone_positions is not None:
            drone_velocities = np.linalg.norm(drones - self.previous_drone_positions, axis=1)
            avg_speed = np.mean(drone_velocities)
            if avg_speed > 0.1:
                patrol_reward = 0.05 * reward_multiplier
                total_reward += patrol_reward
        self.previous_drone_positions = drones.copy()
        info["patrol_reward"] = patrol_reward

        # =========================
        # 2. Couverture de la zone
        # =========================
        coverage_reward = 0.0
        coverage_ratio = 0.0
        new_coverage_bonus = 0.0
        if stage_config.get('enable_coverage', False):
            # Calculer la couverture avant mise à jour
            previous_coverage_cells = np.sum(self.coverage_grid > 0)
            
            for drone in drones:
                drone_pos_2d = drone[:2]
                self._update_coverage_grid(drone_pos_2d, drone_id=None)
            
            # Calculer la couverture après mise à jour
            total_coverage_cells = np.sum(self.coverage_grid > 0)
            coverage_ratio = total_coverage_cells / max(self.grid_size ** 2, 1)
            
            # Récompense de base pour couverture totale
            coverage_reward = REWARD_COVERAGE * coverage_ratio * reward_multiplier
            
            # Bonus pour nouvelles zones découvertes (encourage exploration active)
            new_cells = total_coverage_cells - previous_coverage_cells
            if new_cells > 0:
                new_coverage_bonus = 0.5 * new_cells * reward_multiplier  # Bonus par nouvelle cellule
                coverage_reward += new_coverage_bonus
            
            total_reward += coverage_reward
        info["coverage_reward"] = coverage_reward
        info["coverage_ratio"] = coverage_ratio
        info["new_coverage_bonus"] = new_coverage_bonus

        # =========================
        # 3. Séparation entre drones
        # =========================
        separation_reward = 0.0
        separation_penalty = 0.0
        if stage_config.get('enable_separation', False):
            MIN_DRONE_SEPARATION = 5.0
            OPTIMAL_DRONE_SEPARATION = 15.0
            MAX_DRONE_SEPARATION = 30.0
            for i in range(len(drones)):
                for j in range(i + 1, len(drones)):
                    dist = np.linalg.norm(drones[i][:2] - drones[j][:2])
                    if dist < MIN_DRONE_SEPARATION:
                        penalty = REWARD_OVERLAP_PENALTY * self._get_penalty_multiplier('too_close')
                        separation_penalty += penalty
                    elif MIN_DRONE_SEPARATION <= dist <= OPTIMAL_DRONE_SEPARATION:
                        separation_reward += 0.02 * reward_multiplier
                    elif dist > MAX_DRONE_SEPARATION:
                        penalty = 0.02 * self._get_penalty_multiplier('too_far')
                        separation_penalty += penalty
        total_reward += separation_reward + separation_penalty
        info["separation_reward"] = separation_reward
        info["separation_penalty"] = separation_penalty

        # =========================
        # 4. Reste dans la zone
        # =========================
        in_zone_reward = 0.0
        out_of_zone_penalty = 0.0
        if stage_config.get('enable_zone', False):
            for drone in drones:
                x, z = drone[0], drone[2]
                # Hors de la zone
                if x < ZONE_MIN_X or x > ZONE_MAX_X or z < ZONE_MIN_Z or z > ZONE_MAX_Z:
                    penalty = REWARD_OUT_OF_ZONE_PENALTY * self._get_penalty_multiplier('zone')
                    out_of_zone_penalty += penalty
                # Dans zone sûre
                elif (ZONE_MIN_X + ZONE_BOUNDARY_MARGIN <= x <= ZONE_MAX_X - ZONE_BOUNDARY_MARGIN and
                    ZONE_MIN_Z + ZONE_BOUNDARY_MARGIN <= z <= ZONE_MAX_Z - ZONE_BOUNDARY_MARGIN):
                    in_zone_reward += 0.05 * reward_multiplier
        total_reward += in_zone_reward + out_of_zone_penalty
        info["in_zone_reward"] = in_zone_reward
        info["out_of_zone_penalty"] = out_of_zone_penalty

        # =========================
        # 5. Détection et tracking de l'intrus
        # =========================
        detection_reward = 0.0
        tracking_reward = 0.0
        distances = np.linalg.norm(drones - intruder, axis=1)
        min_distance_to_intruder = np.min(distances)
        
        # Détection
        if min_distance_to_intruder < DETECTION_RADIUS:
            if not self.intruder_detected:
                self.intruder_detected = True
                self.detection_step = self.current_step
                detection_reward = REWARD_DETECTION * reward_multiplier
                total_reward += detection_reward
        
        # Tracking
        self.tracking_drones = []
        for i, dist in enumerate(distances):
            if dist < TRACKING_RADIUS:
                self.tracking_drones.append(i)
        if len(self.tracking_drones) > 0:
            tracking_reward = TRACKING_REWARD * len(self.tracking_drones) * reward_multiplier
            total_reward += tracking_reward
        info["detection_reward"] = detection_reward
        info["tracking_reward"] = tracking_reward

        # =========================
        # 6. Capture de l'intrus
        # =========================
        capture_reward = 0.0
        early_capture_penalty = 0.0
        if min_distance_to_intruder < CAPTURE_RADIUS:
            if not self.intruder_captured:
                self.intruder_captured = True
                self.capture_step = self.current_step
                capture_reward = REWARD_CAPTURE * reward_multiplier
                
                # Pénalité pour capture trop rapide (encourage exploration avant capture)
                # Si capture dans les 10 premiers steps, réduire la récompense
                if self.current_step < 10:
                    early_capture_penalty = -0.3 * REWARD_CAPTURE * reward_multiplier  # Pénalité de 30%
                    capture_reward += early_capture_penalty
                # Bonus pour capture après exploration (après 20 steps)
                elif self.current_step > 20:
                    exploration_bonus = 0.1 * REWARD_CAPTURE * reward_multiplier  # Bonus de 10%
                    capture_reward += exploration_bonus
                
                total_reward += capture_reward
        info["capture_reward"] = capture_reward
        info["early_capture_penalty"] = early_capture_penalty

        # =========================
        # 7. Proximité continue (même avant détection)
        # =========================
        proximity_reward = 0.0
        PROXIMITY_CONTINUOUS_RADIUS = 50.0
        if min_distance_to_intruder < PROXIMITY_CONTINUOUS_RADIUS and not self.intruder_captured:
            proximity_bonus = 0.1 * (PROXIMITY_CONTINUOUS_RADIUS - min_distance_to_intruder) / PROXIMITY_CONTINUOUS_RADIUS
            proximity_reward = proximity_bonus * reward_multiplier
            total_reward += proximity_reward
        info["proximity_reward"] = proximity_reward

        # =========================
        # 8. Collisions entre drones
        # =========================
        collision_penalty = 0.0
        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                dist = np.linalg.norm(drones[i] - drones[j])
                if dist < 1.0:  # Collision
                    penalty = REWARD_COLLISION_PENALTY * self._get_penalty_multiplier('collision')
                    collision_penalty += penalty
        total_reward += collision_penalty
        info["collision_penalty"] = collision_penalty

        # =========================
        # 9. Collisions avec obstacles
        # =========================
        obstacle_penalty = 0.0
        if stage_config.get('enable_obstacles', False) and len(self.current_obstacles) > 0:
            for drone in drones:
                for obs in self.current_obstacles:
                    if obs is None or len(obs) < 4:
                        continue
                    obs_pos = np.array([obs[0], obs[1], obs[2]])
                    obs_radius = obs[3]
                    dist = np.linalg.norm(drone - obs_pos)
                    if dist < obs_radius + 0.5:  # Collision avec obstacle
                        penalty = REWARD_COLLISION_PENALTY * self._get_penalty_multiplier('obstacle_collision')
                        obstacle_penalty += penalty
        total_reward += obstacle_penalty
        info["obstacle_penalty"] = obstacle_penalty

        # =========================
        # 10. Pénalité de distance (si activée)
        # =========================
        distance_penalty = 0.0
        if stage_config.get('enable_distance_penalty', False):
            mean_distance = np.mean(distances)
            if mean_distance > 100.0:  # Trop loin
                distance_penalty = REWARD_DISTANCE_PENALTY * mean_distance * self._get_penalty_multiplier('distance')
                total_reward += distance_penalty
        info["distance_penalty"] = distance_penalty

        # =========================
        # 11. Bonus de stage
        # =========================
        stage_bonus = self._get_stage_bonus()
        total_reward += stage_bonus
        info["stage_bonus"] = stage_bonus

        # =========================
        # 12. Scaling final (si activé)
        # =========================
        if REWARD_SCALING_ENABLED:
            total_reward *= REWARD_SCALING_FACTOR

        # =========================
        # 13. Informations supplémentaires
        # =========================
        info.update({
            "mean_distance": float(np.mean(distances)),
            "min_distance_to_intruder": float(min_distance_to_intruder),
            "detected": int(self.intruder_detected),
            "captured": int(self.intruder_captured),
            "coverage_cells": int(np.sum(self.coverage_grid > 0)),
        })

        return total_reward, info


    def _update_coverage_grid(self, drone_pos_2d, drone_id):
        """
        Met à jour la grille de couverture autour de la position du drone.
        Les zones occupées par des obstacles sont exclues de la couverture.
        """
        # Calculer les indices de grille correspondants
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Position du centre de la cellule
                cell_x = ZONE_MIN_X + (i + 0.5) * self.cell_size_x
                cell_z = ZONE_MIN_Z + (j + 0.5) * self.cell_size_z
                cell_pos = np.array([cell_x, cell_z])
                
                # Vérifier si la cellule est dans le rayon de couverture
                distance = np.linalg.norm(drone_pos_2d - cell_pos)
                if distance <= COVERAGE_RADIUS:
                    # Vérifier si la cellule n'est pas occupée par un obstacle
                    is_obstacle = False
                    if hasattr(self, 'current_obstacles') and len(self.current_obstacles) > 0:
                        for obs in self.current_obstacles:
                            if obs is None or len(obs) < 4:
                                continue
                            obs_pos_2d = np.array([obs[0], obs[2]])  # x, z
                            obs_radius = obs[3]
                            if np.linalg.norm(cell_pos - obs_pos_2d) < obs_radius:
                                is_obstacle = True
                                break
                    
                    # Marquer comme couverte seulement si pas d'obstacle
                    if not is_obstacle:
                        self.coverage_grid[i, j] = 1.0
                        self.coverage_count[i, j] += 1

    def _check_done(self, drones, intruder):
        """Vérifie les conditions de terminaison de l'épisode."""
        terminated = False
        truncated = False
        
        # Terminaison si l'intrus est capturé (objectif principal)
        if self.intruder_captured:
            terminated = True
        
        # Terminaison si l'intrus est détecté (ancien comportement, optionnel)
        # if self.intruder_detected:
        #     terminated = True
        
        # Troncature si nombre maximum de steps atteint
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Troncature si un drone sort complètement de la zone (sécurité) - SEULEMENT si la zone est activée
        # Au Stage 0, les drones n'ont aucune contrainte de zone
        # 🎯 RAPPORT 2 : Utilisation de ZONE_TRUNCATION_MARGIN (5.0 au lieu de 10.0 fixe)
        stage_config = self._get_stage_config()
        if stage_config['enable_zone']:
            for drone in drones:
                if (drone[0] < ZONE_MIN_X - ZONE_TRUNCATION_MARGIN or drone[0] > ZONE_MAX_X + ZONE_TRUNCATION_MARGIN or
                    drone[2] < ZONE_MIN_Z - ZONE_TRUNCATION_MARGIN or drone[2] > ZONE_MAX_Z + ZONE_TRUNCATION_MARGIN):
                    truncated = True
                    break
        
        return terminated, truncated

    def _process_obs(self, obs_json):
        """
        Traite les observations en mode OBSERVATION TOTALE (global state).
        
        Chaque agent reçoit :
        - Positions de tous les drones (x, y, z pour chaque drone)
        - Position de l'intrus (x, y, z)
        - Positions des obstacles (x, y, z, radius pour chaque obstacle)
        - État de détection globale (1 si détecté, 0 sinon)
        - Couverture locale par drone (ratio de couverture dans le voisinage)
        
        Note: Pour observation partielle (future amélioration), chaque drone
        ne verrait que son voisinage local et nécessiterait communication.
        """
        drones = np.array([[d["x"], d["y"], d["z"]] for d in obs_json["drones"]])
        intruder = np.array([obs_json["intruder"]["x"], 
                            obs_json["intruder"]["y"], 
                            obs_json["intruder"]["z"]])
        
        # Traiter les obstacles
        obstacles_list = []
        if "obstacles" in obs_json and obs_json["obstacles"] is not None:
            for obs in obs_json["obstacles"]:
                if obs is not None:
                    obstacles_list.append([
                        obs.get("x", 0.0),
                        obs.get("y", 0.0),
                        obs.get("z", 0.0),
                        obs.get("radius", 1.0)
                    ])
        
        # Stocker les obstacles pour les calculs de récompenses
        self.current_obstacles = obstacles_list.copy() if obstacles_list else []
        
        # Pad ou tronquer les obstacles pour avoir un nombre fixe
        obstacles_array = np.zeros((self.max_obstacles, 4), dtype=np.float32)
        for i, obs in enumerate(obstacles_list[:self.max_obstacles]):
            obstacles_array[i] = obs
        
        # État de détection (1 si détecté, 0 sinon)
        detection_state = np.array([1.0 if self.intruder_detected else 0.0])
        
        # Couverture locale : pour chaque drone, calculer le ratio de couverture dans son voisinage
        coverage_local = []
        for drone in drones:
            drone_pos_2d = drone[:2]
            local_coverage = self._get_local_coverage(drone_pos_2d)
            coverage_local.append(local_coverage)
        coverage_local = np.array(coverage_local)
        
        # Concaténer toutes les observations
        obs = np.concatenate([
            drones.flatten(),
            intruder,
            detection_state,
            coverage_local,
            obstacles_array.flatten()
        ]).astype(np.float32)
        
        return obs

    def _get_local_coverage(self, drone_pos_2d):
        """Calcule le ratio de couverture locale autour d'un drone."""
        local_radius = COVERAGE_RADIUS * 2  # Zone locale étendue
        local_cells = 0
        covered_local_cells = 0
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_x = ZONE_MIN_X + (i + 0.5) * self.cell_size_x
                cell_z = ZONE_MIN_Z + (j + 0.5) * self.cell_size_z
                cell_pos = np.array([cell_x, cell_z])
                
                distance = np.linalg.norm(drone_pos_2d - cell_pos)
                if distance <= local_radius:
                    local_cells += 1
                    if self.coverage_grid[i, j] > 0:
                        covered_local_cells += 1
        
        return covered_local_cells / max(local_cells, 1)

    def render(self):
        """Optionnel : visualisation de l'état de patrouille."""
        pass

    def close(self):
        """Ferme proprement la connexion Unity."""
        if hasattr(self.manager, "close"):
            self.manager.close()


