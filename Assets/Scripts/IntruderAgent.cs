using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IntruderAgent : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 3f;       // Vitesse de déplacement de base
    public float changeDirectionTime = 2f;  // Temps avant de changer de direction
    
    // 🎓 Curriculum Learning: Multiplicateur de vitesse (appliqué depuis Python)
    private float speedMultiplier = 1.0f;  // Multiplicateur de vitesse (0.7x, 0.9x, 1.0x selon le stage)
    
    [Header("Behavior")]
    [Tooltip("Probabilité que l'intrus se diriger vers la zone (0-1). 0.3 = 30% de chance")]
    public float tendencyToEnterZone = 0.3f;  // Probabilité de se diriger vers la zone
    
    [Header("Collision Settings")]
    [Tooltip("Rayon de détection de collision pour l'intrus (distance minimale aux obstacles)")]
    public float intruderCollisionRadius = 0.5f;  // Rayon de collision de l'intrus
    
    [Tooltip("Si activé, l'intrus ne peut pas traverser les obstacles")]
    public bool preventObstacleCollision = true;  // Empêcher les collisions avec les obstacles

    private Vector3 moveDirection;
    private float timer;
    private PatrolZone patrolZone;  // Zone de patrouille
    private bool isInZone = false;  // Track si l'intrus est dans la zone
    private List<GameObject> obstacles;  // Liste des obstacles

    void Start()
    {
        // Récupérer la zone de patrouille
        patrolZone = PatrolZone.Instance;
        if (patrolZone == null)
        {
            Debug.LogWarning("[IntruderAgent] No PatrolZone found. Intruder will move freely.");
        }
        
        // Récupérer les obstacles depuis EnvManager
        var envManager = FindObjectOfType<EnvManager>();
        if (envManager != null && envManager.obstaclesStatic != null)
        {
            obstacles = envManager.obstaclesStatic;
            Debug.Log($"[IntruderAgent] Found {obstacles.Count} obstacles.");
        }
        else
        {
            obstacles = new List<GameObject>();
        }
        
        PickNewDirection();
        UpdateZoneStatus();
    }

    void Update()
    {
        // Calculer le mouvement avec le multiplicateur de vitesse (curriculum learning)
        float effectiveSpeed = moveSpeed * speedMultiplier;
        Vector3 movement = moveDirection * effectiveSpeed * Time.deltaTime;
        Vector3 newPosition = transform.position + movement;
        
        // 🎓 Curriculum: Vérifier les collisions avec les obstacles seulement si activés
        // (Les obstacles sont désactivés au Stage 0, donc cette vérification ne s'applique pas)
        if (preventObstacleCollision && obstacles != null && obstacles.Count > 0)
        {
            // Filtrer seulement les obstacles actifs
            var activeObstacles = obstacles.FindAll(obs => obs != null && obs.activeInHierarchy);
            if (activeObstacles.Count > 0)
            {
                newPosition = CheckObstacleCollision(transform.position, newPosition, activeObstacles);
            }
        }
        
        // Appliquer le mouvement (après vérification des collisions)
        transform.position = newPosition;

        // Timer pour changer de direction
        timer -= Time.deltaTime;
        if (timer <= 0)
        {
            PickNewDirection();
        }
        
        // Mettre à jour le statut (dans/hors zone)
        UpdateZoneStatus();
    }
    
    void UpdateZoneStatus()
    {
        if (patrolZone != null)
        {
            bool wasInZone = isInZone;
            isInZone = patrolZone.IsInZone(transform.position);
            
            if (!wasInZone && isInZone)
            {
                Debug.Log("[IntruderAgent] Intruder ENTERED the patrol zone!");
            }
            else if (wasInZone && !isInZone)
            {
                Debug.Log("[IntruderAgent] Intruder LEFT the patrol zone.");
            }
        }
    }

    void PickNewDirection()
    {
        // Comportement : parfois se diriger vers la zone, parfois aléatoire
        if (patrolZone != null && Random.value < tendencyToEnterZone && !isInZone)
        {
            // Se diriger vers le centre de la zone
            Vector3 center = patrolZone.Center;
            Vector3 directionToZone = (center - transform.position).normalized;
            
            // Ajouter un peu de random pour que ce ne soit pas trop direct
            Vector3 randomComponent = new Vector3(
                Random.Range(-0.5f, 0.5f),
                0,
                Random.Range(-0.5f, 0.5f)
            ).normalized;
            
            moveDirection = (directionToZone + randomComponent * 0.3f).normalized;
        }
        else
        {
            // Direction complètement aléatoire
            moveDirection = new Vector3(Random.Range(-1f, 1f), 0, Random.Range(-1f, 1f)).normalized;
        }
        
        timer = changeDirectionTime;
    }
    
    /// <summary>
    /// Retourne true si l'intrus est actuellement dans la zone de patrouille.
    /// </summary>
    public bool IsInZone()
    {
        return isInZone;
    }
    
    /// <summary>
    /// 🎓 Curriculum Learning: Définit le multiplicateur de vitesse de l'intrus.
    /// Appelé depuis UnityComms lors du reset avec les paramètres du stage.
    /// </summary>
    public void SetSpeedMultiplier(float multiplier)
    {
        speedMultiplier = multiplier;
        Debug.Log($"[IntruderAgent] Speed multiplier set to {multiplier}x (effective speed: {moveSpeed * multiplier})");
    }
    
    /// <summary>
    /// Vérifie si le mouvement proposé causerait une collision avec un obstacle.
    /// Retourne la position corrigée (sans collision) ou la position originale si collision détectée.
    /// </summary>
    private Vector3 CheckObstacleCollision(Vector3 currentPos, Vector3 newPos, List<GameObject> obstaclesToCheck)
    {
        // Utiliser la liste fournie ou la liste globale
        var obstaclesList = obstaclesToCheck ?? obstacles;
        
        if (obstaclesList == null || obstaclesList.Count == 0)
            return newPos;
        
        // Vérifier chaque obstacle
        foreach (var obstacle in obstaclesList)
        {
            if (obstacle == null || !obstacle.activeInHierarchy)
                continue;
            
            Vector3 obstaclePos = obstacle.transform.position;
            Vector3 obstacleScale = obstacle.transform.localScale;
            
            // Obtenir le rayon de l'obstacle depuis son collider
            float obstacleRadius = 1.0f;
            Collider col = obstacle.GetComponent<Collider>();
            if (col != null)
            {
                if (col is BoxCollider boxCol)
                {
                    obstacleRadius = Mathf.Max(boxCol.size.x, boxCol.size.z) * 0.5f;
                }
                else if (col is SphereCollider sphereCol)
                {
                    obstacleRadius = sphereCol.radius;
                }
                else if (col is CapsuleCollider capsuleCol)
                {
                    obstacleRadius = capsuleCol.radius;
                }
            }
            
            // Appliquer le scale
            obstacleRadius *= Mathf.Max(obstacleScale.x, obstacleScale.z);
            
            // Distance minimale requise (rayon obstacle + rayon intrus + petite marge)
            float minDistance = obstacleRadius + intruderCollisionRadius + 0.1f;
            
            // Vérifier la distance entre la nouvelle position et l'obstacle
            float distanceToObstacle = Vector3.Distance(new Vector3(newPos.x, 0, newPos.z), 
                                                         new Vector3(obstaclePos.x, 0, obstaclePos.z));
            
            if (distanceToObstacle < minDistance)
            {
                // Collision détectée ! Empêcher le mouvement
                // Calculer la direction depuis la position actuelle vers l'obstacle
                Vector3 currentPos2D = new Vector3(currentPos.x, 0, currentPos.z);
                Vector3 obstaclePos2D = new Vector3(obstaclePos.x, 0, obstaclePos.z);
                Vector3 directionFromObstacle = (currentPos2D - obstaclePos2D).normalized;
                
                // Si l'intrus est exactement sur l'obstacle (distance = 0), utiliser une direction par défaut
                if (directionFromObstacle.magnitude < 0.01f)
                {
                    directionFromObstacle = Vector3.right; // Direction par défaut
                }
                
                // Position sécurisée : à la distance minimale de l'obstacle, dans la direction opposée
                float safeDistance = minDistance - 0.05f; // Petite marge de sécurité
                Vector3 safePosition = obstaclePos2D + directionFromObstacle * safeDistance;
                
                // Garder la hauteur Y originale
                safePosition.y = newPos.y;
                
                // Retourner la position sécurisée (juste à côté de l'obstacle, mais pas dedans)
                // Si collision, changer de direction pour éviter de rester bloqué
                PickNewDirection();
                return safePosition;
            }
        }
        
        // Aucune collision détectée, mouvement autorisé
        return newPos;
    }
    
    /// <summary>
    /// Retourne le multiplicateur de vitesse actuel (pour debugging).
    /// </summary>
    public float GetSpeedMultiplier()
    {
        return speedMultiplier;
    }
}
