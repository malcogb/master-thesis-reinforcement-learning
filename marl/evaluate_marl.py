# evaluate_marl.py
import os
import sys
import argparse
import glob
import json
import datetime
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au PYTHONPATH pour permettre les imports
# Cela permet de lancer le script depuis n'importe quel répertoire
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
from python.aero_patrol_wrapper import AeroPatrolWrapper
from python.config import SAVE_DIR, MODEL_NAME, MAX_STEPS_PER_EPISODE, NUM_DRONES, REWARD_COLLISION_PENALTY, EVALUATION_LOG_DIR

def list_available_models():
    """Liste tous les modèles disponibles."""
    # SAVE_DIR est relatif au répertoire marl/
    models_dir = os.path.join(current_dir, SAVE_DIR)
    models_path = os.path.join(models_dir, "*.zip")
    models = glob.glob(models_path)
    if not models:
        print("❌ Aucun modèle trouvé.")
        print(f"   Recherche dans : {models_dir}")
        return []
    
    print("📋 Modèles disponibles :")
    for i, model_path in enumerate(sorted(models, reverse=True), 1):
        model_name = os.path.basename(model_path)
        print(f"  {i}. {model_name}")
    return models

def ensure_dir(dir_path):
    """Crée un répertoire s'il n'existe pas."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def save_evaluation_results(results, model_path, output_file=None):
    """
    Sauvegarde les résultats d'évaluation dans un fichier JSON et texte.
    
    Args:
        results: Dictionnaire contenant les résultats de l'évaluation
        model_path: Chemin vers le modèle évalué
        output_file: Chemin vers le fichier de sortie (si None, génère un nom avec timestamp)
    """
    # Créer le répertoire de logs s'il n'existe pas
    evaluation_log_dir = os.path.join(current_dir, EVALUATION_LOG_DIR)
    ensure_dir(evaluation_log_dir)
    
    # Générer un nom de fichier avec timestamp si non spécifié
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path).replace(".zip", "")
        output_file = os.path.join(evaluation_log_dir, f"evaluation_{model_name}_{timestamp}")
    else:
        # Si le chemin est relatif, le considérer comme relatif à evaluation_log_dir
        if not os.path.isabs(output_file):
            output_file = os.path.join(evaluation_log_dir, output_file)
    
    # Sauvegarder en JSON
    json_file = output_file + ".json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Résultats sauvegardés en JSON : {json_file}")
    
    # Sauvegarder en texte lisible
    txt_file = output_file + ".txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("📊 RÉSULTATS DE L'ÉVALUATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"📅 Date : {results['timestamp']}\n")
        f.write(f"🤖 Modèle : {results['model_name']}\n")
        f.write(f"🎯 Épisodes évalués : {results['num_episodes']}\n")
        f.write(f"💰 Récompense moyenne : {results['avg_reward']:.2f}\n")
        f.write(f"📈 Récompense min/max : {results['min_reward']:.2f} / {results['max_reward']:.2f}\n")
        f.write(f"📏 Longueur moyenne des épisodes : {results['avg_episode_length']:.1f} steps\n")
        f.write(f"🔍 Taux de détection : {results['detection_rate']:.1f}% ({results['detections']}/{results['num_episodes']})\n")
        f.write(f"🎯 Taux de capture : {results['capture_rate']:.1f}% ({results['captures']}/{results['num_episodes']})\n")
        f.write(f"💥 Collisions moyennes : {results['avg_collisions']:.2f}\n")
        f.write(f"🚫 Sorties de zone moyennes : {results['avg_out_of_zone']:.2f}\n")
        f.write("=" * 60 + "\n")
        
        # Détails par épisode
        f.write("\n📋 Détails par épisode :\n")
        f.write("-" * 60 + "\n")
        for i, episode in enumerate(results['episodes'], 1):
            status = "✅" if episode['captured'] else ("🔍" if episode['detected'] else "❌")
            f.write(f"Épisode {i}: Reward: {episode['reward']:.2f}, "
                   f"Steps: {episode['steps']}, "
                   f"Collisions: {episode['collisions']:.2f}, "
                   f"Status: {status}\n")
    print(f"💾 Résultats sauvegardés en texte : {txt_file}")

def evaluate(model_path=None, num_episodes=10, verbose=True, save_logs=True, output_file=None):
    """
    Évalue un modèle entraîné.
    
    Args:
        model_path: Chemin vers le modèle (si None, utilise le dernier modèle)
        num_episodes: Nombre d'épisodes pour l'évaluation
        verbose: Afficher les détails
        save_logs: Sauvegarder les résultats dans un fichier
        output_file: Chemin vers le fichier de sortie (si None, génère un nom avec timestamp)
    """
    # Déterminer le modèle à évaluer
    # SAVE_DIR est relatif au répertoire marl/
    models_dir = os.path.join(current_dir, SAVE_DIR)
    
    if model_path is None:
        # Utiliser le modèle standard (dernier entraîné)
        model_path = os.path.join(models_dir, MODEL_NAME + ".zip")
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}.")
            print("📋 Liste des modèles disponibles :")
            list_available_models()
            return
    elif not os.path.isabs(model_path):
        # Si le chemin est relatif, le considérer comme relatif à models_dir
        model_path = os.path.join(models_dir, model_path)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}.")
        print(f"   Recherche dans : {models_dir}")
        list_available_models()
        return
    
    print(f"📦 Chargement du modèle : {os.path.basename(model_path)}")
    
    # Initialise l'environnement
    env = AeroPatrolWrapper(num_drones=NUM_DRONES)
    
    # ⚠️ IMPORTANT : Désactiver la progression du curriculum pendant l'évaluation
    # L'évaluation doit se faire dans des conditions fixes (même stage)
    env.curriculum_enabled = False  # Désactiver la progression automatique
    
    # Tenter de restaurer l'état du curriculum depuis le fichier associé
    curriculum_state_path = model_path.replace('.zip', '_curriculum_state.json')
    if os.path.exists(curriculum_state_path):
        try:
            with open(curriculum_state_path, 'r') as f:
                curriculum_state = json.load(f)
            # Restaurer l'état du curriculum (stage au moment de la sauvegarde)
            env.current_stage = curriculum_state.get('current_stage', 2)  # Par défaut Stage 2 si non trouvé
            print(f"✅ État du curriculum restauré : Stage {env.current_stage}")
            print(f"   → Curriculum désactivé pour l'évaluation (pas de progression)")
        except Exception as e:
            print(f"⚠️  Impossible de restaurer l'état du curriculum : {e}")
            print(f"   → Utilisation du Stage 2 par défaut pour l'évaluation")
            env.current_stage = 2  # Forcer Stage 2 pour l'évaluation
    else:
        print(f"⚠️  Fichier d'état du curriculum non trouvé : {curriculum_state_path}")
        print(f"   → Utilisation du Stage 2 par défaut pour l'évaluation")
        env.current_stage = 2  # Forcer Stage 2 pour l'évaluation
    
    # Charge le modèle PPO
    try:
        model = PPO.load(model_path, env=env)
        print("✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return
    
    # Statistiques d'évaluation
    total_rewards = []
    episode_lengths = []
    detections = []
    captures = []
    collisions = []
    out_of_zone_count = []
    episode_details = []  # Détails par épisode pour la sauvegarde
    
    print(f"\n🎯 Évaluation sur {num_episodes} épisodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        # Reset de l'environnement
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_detected = False
        episode_captured = False
        episode_collisions = 0
        episode_out_of_zone = 0
        
        while not done and episode_steps < MAX_STEPS_PER_EPISODE:
            # Politique déterministe pour l'évaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Collecter les informations de l'épisode
            if info:
                # Info peut être un dict ou une liste (selon l'environnement)
                info_dict = info
                if isinstance(info, list) and len(info) > 0:
                    # Si info est une liste (VecEnv), prendre le premier élément
                    info_dict = info[0]
                
                if isinstance(info_dict, dict):
                    if info_dict.get('intruder_detected', False):
                        episode_detected = True
                    if info_dict.get('intruder_captured', False):
                        episode_captured = True
                    # Compter le nombre réel de collisions (nouvelle métrique)
                    collision_count = info_dict.get('collision_count', 0)
                    if collision_count > 0:
                        episode_collisions += collision_count
                    # Fallback : utiliser collision_penalty si collision_count n'est pas disponible
                    elif 'collision_count' not in info_dict:
                        collision_val = info_dict.get('collision_penalty', 0)
                        if collision_val < 0:
                            # Approximation : diviser par la pénalité pour obtenir le nombre de collisions
                            penalty_value = abs(REWARD_COLLISION_PENALTY)
                            if penalty_value > 0:
                                episode_collisions += abs(collision_val) / penalty_value
                    out_of_zone_val = info_dict.get('out_of_zone_penalty', 0)
                    if out_of_zone_val < 0:
                        episode_out_of_zone += abs(out_of_zone_val)
        
        # Enregistrer les statistiques
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        detections.append(1 if episode_detected else 0)
        captures.append(1 if episode_captured else 0)
        collisions.append(episode_collisions)
        out_of_zone_count.append(episode_out_of_zone)
        
        # Enregistrer les détails de l'épisode
        episode_details.append({
            "episode": episode + 1,
            "reward": float(episode_reward),
            "steps": episode_steps,
            "detected": episode_detected,
            "captured": episode_captured,
            "collisions": float(episode_collisions),
            "out_of_zone": float(episode_out_of_zone)
        })
        
        if verbose:
            status = "✅" if episode_captured else ("🔍" if episode_detected else "❌")
            print(f"Épisode {episode + 1}/{num_episodes}: "
                  f"Reward: {episode_reward:.2f}, "
                  f"Steps: {episode_steps}, "
                  f"Status: {status}")
    
    # Calculer les statistiques
    avg_reward = sum(total_rewards) / len(total_rewards)
    min_reward = min(total_rewards)
    max_reward = max(total_rewards)
    avg_episode_length = sum(episode_lengths) / len(episode_lengths)
    detection_rate = sum(detections) / len(detections) * 100
    capture_rate = sum(captures) / len(captures) * 100
    avg_collisions = sum(collisions) / len(collisions)
    avg_out_of_zone = sum(out_of_zone_count) / len(out_of_zone_count)
    
    # Afficher les résultats
    print("=" * 60)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 60)
    print(f"🎯 Épisodes évalués : {num_episodes}")
    print(f"💰 Récompense moyenne : {avg_reward:.2f}")
    print(f"📈 Récompense min/max : {min_reward:.2f} / {max_reward:.2f}")
    print(f"📏 Longueur moyenne des épisodes : {avg_episode_length:.1f} steps")
    print(f"🔍 Taux de détection : {detection_rate:.1f}% ({sum(detections)}/{num_episodes})")
    print(f"🎯 Taux de capture : {capture_rate:.1f}% ({sum(captures)}/{num_episodes})")
    print(f"💥 Collisions moyennes : {avg_collisions:.2f}")
    print(f"🚫 Sorties de zone moyennes : {avg_out_of_zone:.2f}")
    print("=" * 60)
    
    # Sauvegarder les résultats si demandé
    if save_logs:
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_name": os.path.basename(model_path),
            "model_path": model_path,
            "num_episodes": num_episodes,
            "avg_reward": float(avg_reward),
            "min_reward": float(min_reward),
            "max_reward": float(max_reward),
            "avg_episode_length": float(avg_episode_length),
            "detection_rate": float(detection_rate),
            "capture_rate": float(capture_rate),
            "detections": sum(detections),
            "captures": sum(captures),
            "avg_collisions": float(avg_collisions),
            "avg_out_of_zone": float(avg_out_of_zone),
            "episodes": episode_details
        }
        save_evaluation_results(results, model_path, output_file)

        # Sauvegarder une courbe des rewards par épisode (moyenne + IC95)
        # Ajout d'un suffixe horodaté pour ne pas écraser les anciennes images
        ts_rewards = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_rewards = output_file or os.path.join(
            os.path.join(current_dir, EVALUATION_LOG_DIR),
            f"evaluation_{os.path.basename(model_path).replace('.zip','')}_{ts_rewards}"
        )
        rewards_plot_path = base_rewards + f"_rewards_{ts_rewards}.png"
        try:
            mean_r = avg_reward
            std_r = float(np.std(total_rewards)) if total_rewards else 0.0
            n = max(len(total_rewards), 1)
            ic95 = 1.96 * std_r / np.sqrt(n)
            x = range(1, num_episodes + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(x, total_rewards, label="Reward par épisode")
            plt.axhline(mean_r, color="red", linestyle="--", label=f"Moyenne = {mean_r:.2f}")
            # Bande moyenne ± IC95
            plt.fill_between(x, mean_r - ic95, mean_r + ic95, color="red", alpha=0.12, label=f"IC95 ± {ic95:.2f}")
            plt.xlabel("Épisode")
            plt.ylabel("Reward")
            plt.title("Rewards par épisode (moyenne ± IC95)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(rewards_plot_path)
            plt.close()
            print(f"📈 Courbe des rewards sauvegardée : {rewards_plot_path}")
        except Exception as e:
            print(f"⚠️  Impossible de sauvegarder la courbe des rewards : {e}")
    
    # Fermer l'environnement
    env.close()
    
    return results if save_logs else None


def log_to_tensorboard(writer: SummaryWriter, run_idx: int, results: Dict[str, Any]) -> None:
    """Log des métriques principales d'une évaluation unique dans TensorBoard."""
    writer.add_scalar("eval/reward_mean", results["avg_reward"], run_idx)
    writer.add_scalar("eval/reward_min", results["min_reward"], run_idx)
    writer.add_scalar("eval/reward_max", results["max_reward"], run_idx)
    writer.add_scalar("eval/episode_length_mean", results["avg_episode_length"], run_idx)
    writer.add_scalar("eval/detection_rate", results["detection_rate"], run_idx)
    writer.add_scalar("eval/capture_rate", results["capture_rate"], run_idx)
    writer.add_scalar("eval/collisions_mean", results["avg_collisions"], run_idx)
    writer.add_scalar("eval/out_of_zone_mean", results["avg_out_of_zone"], run_idx)


def aggregate_runs(results_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calcule moyenne et écart-type sur plusieurs évaluations."""
    keys = [
        "avg_reward",
        "min_reward",
        "max_reward",
        "avg_episode_length",
        "detection_rate",
        "capture_rate",
        "avg_collisions",
        "avg_out_of_zone",
    ]
    agg: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [r[k] for r in results_list]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def plot_mean_std(agg: Dict[str, Dict[str, float]], output_path: str):
    """Trace un barplot des moyennes avec barres d'erreur (std) et retourne la figure."""
    # S'assurer que le dossier existe (évite les erreurs d'écriture)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    metrics_to_plot = [
        ("avg_reward", "Récompense"),
        ("avg_episode_length", "Longueur épisode"),
        ("avg_collisions", "Collisions"),
        ("avg_out_of_zone", "Sorties de zone"),
    ]
    labels = [m[1] for m in metrics_to_plot]
    means = [agg[m[0]]["mean"] for m in metrics_to_plot]
    stds = [agg[m[0]]["std"] for m in metrics_to_plot]

    fig = plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=6, color="#4c72b0")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Valeur")
    plt.title("Évaluations multi-runs : moyenne ± écart-type")
    plt.tight_layout()
    fig.savefig(output_path)
    print(f"📈 Plot sauvegardé : {output_path}")
    return fig


def evaluate_multiple_runs(
    model_path: str,
    num_runs: int,
    num_episodes: int,
    verbose: bool,
    save_logs: bool,
    output_prefix: str,
    tb_logdir: str,
) -> None:
    """Boucle sur plusieurs évaluations et journalise dans TensorBoard + plot."""
    ensure_dir(tb_logdir)
    writer = SummaryWriter(tb_logdir)

    all_results: List[Dict[str, Any]] = []

    for run_idx in range(num_runs):
        print(f"\n🚀 Run d'évaluation {run_idx + 1}/{num_runs}")
        # Donner un suffixe de fichier différent par run si l'utilisateur a fourni un output_prefix
        run_output = None
        if output_prefix:
            run_output = f"{output_prefix}_run{run_idx + 1}"

        res = evaluate(
            model_path=model_path,
            num_episodes=num_episodes,
            verbose=verbose,
            save_logs=save_logs,
            output_file=run_output,
        )
        if res:
            all_results.append(res)
            log_to_tensorboard(writer, run_idx, res)

    writer.flush()
    writer.close()

    if not all_results:
        print("❌ Aucune évaluation n'a produit de résultats.")
        return

    agg = aggregate_runs(all_results)

    print("\n📊 Synthèse multi-runs (moyenne ± écart-type) :")
    print(f"- Récompense moyenne : {agg['avg_reward']['mean']:.2f} ± {agg['avg_reward']['std']:.2f}")
    print(f"- Longueur épisode : {agg['avg_episode_length']['mean']:.2f} ± {agg['avg_episode_length']['std']:.2f}")
    print(f"- Collisions : {agg['avg_collisions']['mean']:.2f} ± {agg['avg_collisions']['std']:.2f}")
    print(f"- Sorties de zone : {agg['avg_out_of_zone']['mean']:.2f} ± {agg['avg_out_of_zone']['std']:.2f}")
    print(f"- Détection : {agg['detection_rate']['mean']:.1f}% ± {agg['detection_rate']['std']:.1f}%")
    print(f"- Capture : {agg['capture_rate']['mean']:.1f}% ± {agg['capture_rate']['std']:.1f}%")

    # Plot
    plot_path = os.path.join(tb_logdir, "eval_multi_runs_mean_std.png")
    fig = plot_mean_std(agg, plot_path)

    # Journaliser le barplot (moyenne ± std) dans TensorBoard/Images
    writer_img = SummaryWriter(os.path.join(tb_logdir, "images"))
    writer_img.add_figure("eval/mean_std_barplot", fig, 0)
    writer_img.flush()
    writer_img.close()
    plt.close(fig)

    # Journaliser les agrégats dans TensorBoard
    writer = SummaryWriter(os.path.join(tb_logdir, "aggregate"))
    for k, vals in agg.items():
        writer.add_scalar(f"aggregate/{k}_mean", vals["mean"], 0)
        writer.add_scalar(f"aggregate/{k}_std", vals["std"], 0)
    writer.flush()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un modèle PPO entraîné")
    parser.add_argument("--model", type=str, default=None, 
                       help="Chemin vers le modèle (si non spécifié, utilise le dernier modèle)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Nombre d'épisodes pour l'évaluation (défaut: 10)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Nombre de runs d'évaluation à enchaîner (défaut: 1)")
    parser.add_argument("--list", action="store_true",
                       help="Liste les modèles disponibles")
    parser.add_argument("--no-save", action="store_true",
                       help="Ne pas sauvegarder les résultats dans un fichier")
    parser.add_argument("--output", type=str, default=None,
                       help="Chemin vers le fichier de sortie (si non spécifié, génère un nom avec timestamp)")
    parser.add_argument("--tb-logdir", type=str, default=None,
                       help="Répertoire TensorBoard pour les évaluations multi-runs (défaut: logs/tensorboard/eval_multi)")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    else:
        if args.runs > 1:
            # Pour éviter d'écraser d'anciens logs, on date le répertoire par défaut
            if args.tb_logdir:
                tb_dir = args.tb_logdir
            else:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                tb_dir = os.path.join(current_dir, "logs", "tensorboard", f"eval_multi_{ts}")
            ensure_dir(tb_dir)
            evaluate_multiple_runs(
                model_path=args.model,
                num_runs=args.runs,
                num_episodes=args.episodes,
                verbose=True,
                save_logs=not args.no_save,
                output_prefix=args.output,
                tb_logdir=tb_dir,
            )
        else:
            evaluate(model_path=args.model, 
                    num_episodes=args.episodes, 
                    save_logs=not args.no_save,
                    output_file=args.output)
