using UnityEngine;
using System.Collections.Generic;
using PeacefulPie;

public class EnvManager : MonoBehaviour
{
    [Header("Agents dans la scène")]
    [Tooltip("⚠️ Les drones seront créés dynamiquement. Cette liste sera remplie automatiquement.")]
    public List<GameObject> dronesStatic;
    public GameObject intruderStatic;
    
    [Header("Création Dynamique")]
    [Tooltip("Prefab du drone à instancier (doit avoir le script DroneAgent)")]
    public GameObject dronePrefab;
    [Tooltip("Nombre de drones à créer dynamiquement au démarrage (par défaut: 3)")]
    public int numDrones = 3;
    
    [Header("Obstacles")]
    [Tooltip("Liste des obstacles dans la scène (bâtiments, arbres, etc.)")]
    public List<GameObject> obstaclesStatic;

    [Header("Settings")]
    [Tooltip("Si non défini, utilise PatrolZone.Instance pour la zone de spawn")]
    public Vector3 spawnArea = new Vector3(50f, 1f, 50f);  // Zone de spawn: -50 à +50 en X et Z

    private List<GameObject> drones => dronesStatic;
    private PatrolZone patrolZone;
    private bool isFirstReset = true;  // 🚁 Flag pour savoir si c'est le premier reset (garder position initiale)
    private int currentStage = 0;  // 🎓 Stage actuel du curriculum (0, 1, ou 2)
    
    // 🎓 Tailles d'espaces par stage (pour exploration progressive)
    private const float STAGE0_SPACE_SIZE = 100f;  // Stage 0: Espace pour exploration (100×100) - Réduit de 150 à 100
    private const float STAGE1_SPACE_SIZE = 100f;  // Stage 1+: Zone de défense (100×100)
    
    /// <summary>
    /// 🚁 Crée les drones dynamiquement à partir du prefab au démarrage (Play).
    /// Les drones sont créés une seule fois au démarrage et ne seront plus modifiés pendant l'entraînement.
    /// </summary>
    public void CreateDronesDynamically()
    {
        // Vérifier si les drones existent déjà (créés manuellement ou précédemment)
        if (dronesStatic != null && dronesStatic.Count > 0)
        {
            // Filtrer les drones null (peut arriver si supprimés)
            dronesStatic.RemoveAll(d => d == null);
            
            if (dronesStatic.Count > 0)
            {
                Debug.Log($"[EnvManager] {dronesStatic.Count} drones déjà présents dans la scène. Création dynamique ignorée.");
                return;
            }
        }
        
        // Initialiser la liste si null
        if (dronesStatic == null)
        {
            dronesStatic = new List<GameObject>();
        }
        
        // Vérifier que le prefab est assigné
        if (dronePrefab == null)
        {
            Debug.LogError("[EnvManager] Drone prefab non assigné ! Impossible de créer les drones dynamiquement.");
            Debug.LogError("[EnvManager] Veuillez assigner le prefab 'Drone' dans l'inspecteur de EnvManager.");
            return;
        }
        
        // Créer les drones au démarrage (Play)
        Debug.Log($"[EnvManager] Création dynamique de {numDrones} drones au démarrage (Play)...");
        
        // Déterminer la zone de spawn visible
        // Utiliser la zone de patrouille comme référence pour un positionnement visible
        // même si elle est désactivée au Stage 0 (juste pour la visibilité initiale)
        float visibleAreaX = 50f;  // Zone visible par défaut
        float visibleAreaZ = 50f;
        float spawnY = 1f;  // Hauteur de spawn
        
        if (patrolZone != null)
        {
            // Utiliser la zone de patrouille comme référence pour un positionnement visible
            visibleAreaX = patrolZone.zoneSizeX;
            visibleAreaZ = patrolZone.zoneSizeZ;
            spawnY = patrolZone.zoneY;
            Debug.Log($"[EnvManager] Utilisation de la zone de patrouille comme référence pour positionnement visible: {visibleAreaX}x{visibleAreaZ}");
        }
        
        for (int i = 0; i < numDrones; i++)
        {
            // Instancier le drone depuis le prefab
            GameObject drone = Instantiate(dronePrefab);
            drone.name = $"Drone_{i + 1}";
            
            // Positionner le drone dans une zone visible centrée autour de l'origine
            // Cette position est juste pour la visibilité dans la scène Unity
            // Le premier reset depuis Python repositionnera les drones selon le stage
            Vector3 spawnPos = new Vector3(
                Random.Range(-visibleAreaX, visibleAreaX),
                spawnY + Random.Range(0f, 1f),  // Légère variation de hauteur
                Random.Range(-visibleAreaZ, visibleAreaZ)
            );
            drone.transform.position = spawnPos;
            
            // Ajouter à la liste
            dronesStatic.Add(drone);
            
            Debug.Log($"[EnvManager] Drone {i + 1} créé à la position visible {spawnPos} (zone: {visibleAreaX}x{visibleAreaZ})");
        }
        
        Debug.Log($"[EnvManager] ✅ {numDrones} drones créés dynamiquement avec succès ! (Création au démarrage)");
    }

    void Start()
    {
        // Récupérer la zone de patrouille AVANT de créer les drones
        // (nécessaire pour positionner les drones de manière visible)
        patrolZone = PatrolZone.Instance;
        if (patrolZone != null)
        {
            // Utiliser la zone de patrouille pour le spawn
            spawnArea = new Vector3(patrolZone.zoneSizeX, patrolZone.zoneY, patrolZone.zoneSizeZ);
            Debug.Log($"[EnvManager] Using PatrolZone for spawn area: {spawnArea}");
        }
        else
        {
            Debug.LogWarning("[EnvManager] No PatrolZone found. Using default spawnArea.");
        }
        
        // 🚁 CRÉATION DYNAMIQUE DES DRONES : Création au démarrage (Play)
        // Les drones sont créés immédiatement au démarrage de Unity
        // et positionnés de manière visible dans la scène
        CreateDronesDynamically();
        
        // Auto-détection des obstacles si non assignés
        if (obstaclesStatic == null || obstaclesStatic.Count == 0)
        {
            var obstacleManager = FindObjectOfType<ObstacleManager>();
            if (obstacleManager != null)
            {
                obstaclesStatic = obstacleManager.GetObstacles();
                if (obstaclesStatic != null && obstaclesStatic.Count > 0)
                {
                    Debug.Log($"[EnvManager] Auto-found {obstaclesStatic.Count} obstacles from ObstacleManager.");
                }
            }
        }
        
        if (intruderStatic == null)
            Debug.LogWarning("No intruder assigned!");

        // Positionner l'intrus à l'extérieur de la zone au démarrage
        // (même si ResetEnv() n'a pas encore été appelé depuis Python)
        if (intruderStatic != null)
        {
            Vector3 pos;
            if (patrolZone != null)
            {
                // Vérifier si l'intrus est déjà dans la zone
                if (patrolZone.IsInZone(intruderStatic.transform.position))
                {
                    // Repositionner à l'extérieur
                    pos = patrolZone.GetRandomPositionOutsideZone(margin: 10f);
                    intruderStatic.transform.position = pos;
                    Debug.Log($"[EnvManager] Intruder repositioned OUTSIDE zone at {pos} (was inside zone)");
                }
                else
                {
                    Debug.Log($"[EnvManager] Intruder already outside zone at {intruderStatic.transform.position}");
                }
            }
            else
            {
                // Fallback : positionner juste à l'extérieur
                float margin = 10f;
                float side = Random.Range(0, 4); // Choisir un côté aléatoire
                switch (side)
                {
                    case 0: // Nord (Z+)
                        pos = new Vector3(
                            Random.Range(-spawnArea.x, spawnArea.x),
                            spawnArea.y,
                            spawnArea.z + margin
                        );
                        break;
                    case 1: // Sud (Z-)
                        pos = new Vector3(
                            Random.Range(-spawnArea.x, spawnArea.x),
                            spawnArea.y,
                            -spawnArea.z - margin
                        );
                        break;
                    case 2: // Est (X+)
                        pos = new Vector3(
                            spawnArea.x + margin,
                            spawnArea.y,
                            Random.Range(-spawnArea.z, spawnArea.z)
                        );
                        break;
                    default: // Ouest (X-)
                        pos = new Vector3(
                            -spawnArea.x - margin,
                            spawnArea.y,
                            Random.Range(-spawnArea.z, spawnArea.z)
                        );
                        break;
                }
                intruderStatic.transform.position = pos;
                Debug.Log($"[EnvManager] Intruder positioned OUTSIDE zone at {pos}");
            }
        }

        // Lier automatiquement à UnityComms (drones créés au démarrage)
        var comms = FindObjectOfType<UnityComms>();
        if (comms != null)
        {
            // 🚁 Assigner les drones créés au démarrage
            if (dronesStatic != null && dronesStatic.Count > 0)
            {
                comms.AssignDrones(dronesStatic);
            }
            comms.AssignIntruder(intruderStatic);
            if (obstaclesStatic != null && obstaclesStatic.Count > 0)
            {
                comms.AssignObstacles(obstaclesStatic);
            }
        }
    }

    /// <summary>
    /// Réinitialise l'environnement en repositionnant aléatoirement les agents.
    /// Appelé automatiquement lors du reset depuis Python.
    /// </summary>
    /// <param name="stage">Stage actuel du curriculum (0, 1, ou 2). Si non fourni, utilise le stage actuel.</param>
    public void ResetEnv(int? stage = null)
    {
        if (drones == null || drones.Count == 0)
        {
            Debug.LogWarning("[EnvManager] No drones to reset!");
            return;
        }

        // 🎓 Mettre à jour le stage actuel si fourni
        if (stage.HasValue)
        {
            currentStage = stage.Value;
            Debug.Log($"[EnvManager] Stage mis à jour : {currentStage}");
        }

        // 🚁 PREMIER RESET : Garder la position initiale des drones (créés au démarrage)
        if (isFirstReset)
        {
            Debug.Log($"[EnvManager] Premier reset : Les drones gardent leur position initiale de création.");
            Debug.Log($"[EnvManager] Les drones commencent l'entraînement depuis leur position initiale.");
            isFirstReset = false;  // Marquer que le premier reset est fait
            // Ne pas repositionner les drones, ils gardent leur position initiale
        }
        else
        {
            // RESETS SUIVANTS : Repositionner les drones selon le stage
            foreach (var d in drones)
            {
                if (d == null) continue;  // Ignorer les drones null
                
                Vector3 pos;
                // 🎓 Stage 0: Espace pour exploration (100×100) - Aucune contrainte de zone
                // 🎓 Stage 1+: Zone de défense (100×100) - Contraintes activées
                if (patrolZone != null && patrolZone.gameObject.activeInHierarchy)
                {
                    // Zone activée (Stage 1+): positionner les drones DANS la zone de défense (100×100)
                    pos = patrolZone.GetRandomPosition();
                    Debug.Log($"[EnvManager] Drone repositioned IN ZONE at {pos} (Zone activée - Stage {currentStage}, espace: 100×100)");
                }
                else
                {
                    // Zone désactivée (Stage 0): positionnement dans un espace pour exploration
                    // Stage 0 utilise un espace de 100×100 (réduit de 150×150)
                    float spaceSize = currentStage == 0 ? STAGE0_SPACE_SIZE : STAGE1_SPACE_SIZE;
                    pos = new Vector3(
                        Random.Range(-spaceSize, spaceSize),
                        Random.Range(0.5f, 2.0f),  // Hauteur variable
                        Random.Range(-spaceSize, spaceSize)
                    );
                    Debug.Log($"[EnvManager] Drone repositioned OUTSIDE ZONE at {pos} (Stage {currentStage}, espace: {spaceSize * 2}×{spaceSize * 2}, aucune contrainte de zone)");
                }
                d.transform.position = pos;
            }
        }

        // Positionner l'intrus À L'EXTÉRIEUR de la zone de patrouille
        if (intruderStatic != null)
        {
            Vector3 pos;
            if (patrolZone != null)
            {
                // Positionner l'intrus à l'extérieur de la zone (scénario réaliste)
                pos = patrolZone.GetRandomPositionOutsideZone(margin: 10f);
            }
            else
            {
                // Fallback : positionner juste à l'extérieur de la zone
                float margin = 10f;
                float side = Random.Range(0, 4); // Choisir un côté aléatoire
                switch (side)
                {
                    case 0: // Nord (Z+)
                        pos = new Vector3(
                            Random.Range(-spawnArea.x, spawnArea.x),
                            spawnArea.y,
                            spawnArea.z + margin
                        );
                        break;
                    case 1: // Sud (Z-)
                        pos = new Vector3(
                            Random.Range(-spawnArea.x, spawnArea.x),
                            spawnArea.y,
                            -spawnArea.z - margin
                        );
                        break;
                    case 2: // Est (X+)
                        pos = new Vector3(
                            spawnArea.x + margin,
                            spawnArea.y,
                            Random.Range(-spawnArea.z, spawnArea.z)
                        );
                        break;
                    default: // Ouest (X-)
                        pos = new Vector3(
                            -spawnArea.x - margin,
                            spawnArea.y,
                            Random.Range(-spawnArea.z, spawnArea.z)
                        );
                        break;
                }
            }
            intruderStatic.transform.position = pos;
            Debug.Log($"[EnvManager] Intruder spawned OUTSIDE zone at {pos}");
        }

        Debug.Log($"[EnvManager] Environment reset completed. Spawned {drones.Count} drones in area {spawnArea}.");
    }
    
    /// <summary>
    /// Positionne automatiquement tous les agents dans la zone de patrouille.
    /// Peut être appelé depuis l'éditeur Unity (bouton dans l'inspecteur).
    /// Utile pour la visualisation et le debugging.
    /// </summary>
    [ContextMenu("Positionner les agents dans la zone")]
    public void PositionAgentsInZone()
    {
        // Récupérer la zone de patrouille
        if (patrolZone == null)
        {
            patrolZone = PatrolZone.Instance;
        }
        
        if (patrolZone == null)
        {
            Debug.LogWarning("[EnvManager] No PatrolZone found. Using default spawnArea.");
        }
        else
        {
            spawnArea = new Vector3(patrolZone.zoneSizeX, patrolZone.zoneY, patrolZone.zoneSizeZ);
        }

        // Positionner les drones
        if (drones != null && drones.Count > 0)
        {
            int placed = 0;
            foreach (var d in drones)
            {
                if (d == null) continue;
                
                // Position aléatoire dans la zone
                Vector3 newPos = new Vector3(
                    Random.Range(-spawnArea.x, spawnArea.x),
                    spawnArea.y,
                    Random.Range(-spawnArea.z, spawnArea.z)
                );
                
                // Utiliser PatrolZone si disponible pour garantir qu'on est dans la zone
                if (patrolZone != null)
                {
                    newPos = patrolZone.GetRandomPosition();
                }
                
                d.transform.position = newPos;
                placed++;
            }
            Debug.Log($"[EnvManager] {placed} drones positioned in zone (spawnArea: {spawnArea})");
        }
        else
        {
            Debug.LogWarning("[EnvManager] No drones assigned!");
        }

        // Positionner l'intruder À L'EXTÉRIEUR de la zone (scénario réaliste)
        if (intruderStatic != null)
        {
            Vector3 newPos;
            if (patrolZone != null)
            {
                // Positionner à l'extérieur
                newPos = patrolZone.GetRandomPositionOutsideZone(margin: 10f);
            }
            else
            {
                // Fallback : positionner juste à l'extérieur
                float margin = 10f;
                newPos = new Vector3(
                    -spawnArea.x - margin,
                    spawnArea.y,
                    Random.Range(-spawnArea.z, spawnArea.z)
                );
            }
            
            intruderStatic.transform.position = newPos;
            Debug.Log($"[EnvManager] Intruder positioned OUTSIDE zone at {newPos}");
        }
        else
        {
            Debug.LogWarning("[EnvManager] No intruder assigned!");
        }
    }
    
    /// <summary>
    /// Vérifie si tous les agents sont dans la zone de patrouille.
    /// Utile pour le debugging.
    /// </summary>
    [ContextMenu("Vérifier positions des agents")]
    public void CheckAgentPositions()
    {
        if (patrolZone == null)
        {
            patrolZone = PatrolZone.Instance;
        }
        
        if (patrolZone == null)
        {
            Debug.LogWarning("[EnvManager] No PatrolZone found. Cannot check positions.");
            return;
        }

        // Vérifier les drones
        if (drones != null && drones.Count > 0)
        {
            int inZone = 0;
            int outOfZone = 0;
            foreach (var d in drones)
            {
                if (d == null) continue;
                
                if (patrolZone.IsInZone(d.transform.position))
                {
                    inZone++;
                }
                else
                {
                    outOfZone++;
                    Debug.LogWarning($"[EnvManager] Drone '{d.name}' is OUT of zone at {d.transform.position}");
                }
            }
            Debug.Log($"[EnvManager] Drones in zone: {inZone}/{drones.Count}, Out of zone: {outOfZone}");
        }

        // Vérifier l'intruder
        if (intruderStatic != null)
        {
            if (patrolZone.IsInZone(intruderStatic.transform.position))
            {
                Debug.Log($"[EnvManager] Intruder is in zone at {intruderStatic.transform.position}");
            }
            else
            {
                Debug.LogWarning($"[EnvManager] Intruder is OUT of zone at {intruderStatic.transform.position}");
            }
        }
    }
}
