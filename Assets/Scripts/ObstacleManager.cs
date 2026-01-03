using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Script pour gérer automatiquement les obstacles dans la scène.
/// Facilite l'assignation des obstacles à EnvManager.
/// </summary>
public class ObstacleManager : MonoBehaviour
{
    [Header("Auto-détection")]
    [Tooltip("Si activé, trouve automatiquement tous les GameObjects avec le tag 'Obstacle'")]
    public bool autoFindObstacles = true;
    
    [Header("Obstacles Manuels")]
    [Tooltip("Liste manuelle des obstacles (si autoFindObstacles est désactivé)")]
    public List<GameObject> obstaclesList;
    
    [Header("Settings")]
    [Tooltip("Tag utilisé pour identifier les obstacles (si autoFindObstacles est activé)")]
    public string obstacleTag = "Obstacle";
    
    private List<GameObject> obstacles => obstaclesList;

    void Start()
    {
        // Initialiser la liste si elle est null
        if (obstaclesList == null)
        {
            obstaclesList = new List<GameObject>();
        }
        
        // Auto-détection des obstacles
        if (autoFindObstacles)
        {
            // Méthode 1 : Chercher d'abord dans le GameObject "Obstacles" (plus fiable)
            GameObject obstaclesParent = GameObject.Find("Obstacles");
            if (obstaclesParent != null)
            {
                obstaclesList = new List<GameObject>();
                foreach (Transform child in obstaclesParent.transform)
                {
                    if (child.gameObject.activeInHierarchy)
                    {
                        obstaclesList.Add(child.gameObject);
                    }
                }
                if (obstaclesList.Count > 0)
                {
                    Debug.Log($"[ObstacleManager] Found {obstaclesList.Count} obstacles in 'Obstacles' GameObject.");
                }
            }
            
            // Méthode 2 : Si aucun trouvé, essayer par tag (peut échouer si tag n'existe pas)
            if (obstaclesList == null || obstaclesList.Count == 0)
            {
                FindObstaclesByTag();
            }
        }
        
        // Assigner à EnvManager si disponible
        var envManager = FindObjectOfType<EnvManager>();
        if (envManager != null)
        {
            if (obstaclesList != null && obstaclesList.Count > 0)
            {
                envManager.obstaclesStatic = obstaclesList;
                Debug.Log($"[ObstacleManager] Assigned {obstaclesList.Count} obstacles to EnvManager.");
            }
            else
            {
                Debug.Log($"[ObstacleManager] No obstacles found. This is normal if there are no obstacles in the scene.");
                // Initialiser une liste vide pour éviter les erreurs
                envManager.obstaclesStatic = new List<GameObject>();
            }
        }
    }
    
    void FindObstaclesByTag()
    {
        try
        {
            GameObject[] found = GameObject.FindGameObjectsWithTag(obstacleTag);
            if (found != null && found.Length > 0)
            {
                obstaclesList = new List<GameObject>(found);
                Debug.Log($"[ObstacleManager] Found {obstaclesList.Count} obstacles with tag '{obstacleTag}'.");
            }
        }
        catch (UnityException)
        {
            // Le tag n'existe pas dans Unity - ce n'est pas grave, on utilisera la méthode alternative
            Debug.LogWarning($"[ObstacleManager] Tag '{obstacleTag}' is not defined in Unity. Using alternative method to find obstacles.");
        }
    }
    
    /// <summary>
    /// Retourne la liste actuelle des obstacles.
    /// </summary>
    public List<GameObject> GetObstacles()
    {
        return obstaclesList;
    }
}

