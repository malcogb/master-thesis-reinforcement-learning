using UnityEngine;

/// <summary>
/// Script centralisé pour définir la zone de patrouille.
/// Utilisé par tous les agents et l'environnement pour garantir la cohérence.
/// Peut être lié automatiquement à un Plane ou Terrain Unity.
/// </summary>
public class PatrolZone : MonoBehaviour
{
    [Header("Zone de Patrouille")]
    [Tooltip("Option 1: Lier à un Plane/Terrain (recommandé) - la zone utilisera automatiquement la taille du plane")]
    public GameObject targetPlane;  // Plane ou Terrain à utiliser comme zone
    
    [Tooltip("Option 2: Définir manuellement la zone (si targetPlane est null)")]
    public float zoneSizeX = 50f;  // Zone de -zoneSizeX à +zoneSizeX en X
    public float zoneSizeZ = 50f;  // Zone de -zoneSizeZ à +zoneSizeZ en Z
    public float zoneY = 1f;       // Hauteur Y pour le spawn
    
    [Header("Auto-détection")]
    [Tooltip("Si activé, cherche automatiquement un Plane nommé 'Plane' dans la scène")]
    public bool autoFindPlane = true;
    
    [Header("Visualisation (Optionnel)")]
    [Tooltip("Cocher cette case pour voir la zone dans la scène Unity (Gizmos)")]
    public bool showGizmos = true;
    public Color gizmoColor = new Color(0f, 1f, 0f, 0.3f);
    
    // Propriétés pour accès facile
    public float MinX => -zoneSizeX;
    public float MaxX => zoneSizeX;
    public float MinZ => -zoneSizeZ;
    public float MaxZ => zoneSizeZ;
    public Vector3 Center => new Vector3(0, zoneY, 0);
    public Vector3 Size => new Vector3(zoneSizeX * 2, 0, zoneSizeZ * 2);
    
    // Singleton pour accès facile depuis d'autres scripts
    private static PatrolZone _instance;
    public static PatrolZone Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindObjectOfType<PatrolZone>();
                if (_instance == null)
                {
                    Debug.LogWarning("[PatrolZone] No instance found in scene. Creating default zone.");
                    GameObject go = new GameObject("PatrolZone");
                    _instance = go.AddComponent<PatrolZone>();
                }
            }
            return _instance;
        }
    }
    
    void Awake()
    {
        if (_instance == null)
        {
            _instance = this;
        }
        else if (_instance != this)
        {
            Debug.LogWarning("[PatrolZone] Multiple instances found. Keeping the first one.");
            Destroy(this);
            return;
        }
        
        // Auto-détection du plane si activé
        if (autoFindPlane && targetPlane == null)
        {
            GameObject plane = GameObject.Find("Plane");
            if (plane != null)
            {
                targetPlane = plane;
                Debug.Log($"[PatrolZone] Auto-found Plane: {plane.name}");
            }
        }
        
        // Détecter la taille du plane/terrain si assigné
        if (targetPlane != null)
        {
            UpdateZoneFromPlane();
            // 🎓 Stage 0 : Désactiver le Plane dès le début (avant même le premier reset)
            targetPlane.SetActive(false);
            Debug.Log("[PatrolZone] Plane désactivé au démarrage (Stage 0). Sera activé au Stage 1.");
        }
        
        // Note: Le GameObject PatrolZone lui-même sera désactivé depuis UnityComms au démarrage
        // pour éviter de casser le singleton si on le désactive trop tôt
    }
    
    /// <summary>
    /// Met à jour la zone de patrouille en fonction de la taille du Plane/Terrain.
    /// </summary>
    void UpdateZoneFromPlane()
    {
        if (targetPlane == null) return;
        
        // Pour un Plane Unity standard (10x10 unités par défaut, mais peut être scalé)
        Transform planeTransform = targetPlane.transform;
        Vector3 scale = planeTransform.localScale;
        
        // Un Plane Unity standard fait 10x10 unités, mais peut être scalé
        float planeSize = 10f; // Taille par défaut d'un Plane Unity
        zoneSizeX = (planeSize * scale.x) / 2f;  // Diviser par 2 car centré sur l'origine
        zoneSizeZ = (planeSize * scale.z) / 2f;
        
        // Utiliser la hauteur Y du plane
        zoneY = planeTransform.position.y + 0.1f; // Légèrement au-dessus du plane
        
        Debug.Log($"[PatrolZone] Zone updated from Plane: sizeX={zoneSizeX}, sizeZ={zoneSizeZ}, Y={zoneY}");
    }
    
    /// <summary>
    /// Appelé depuis l'éditeur Unity pour mettre à jour la zone en temps réel.
    /// </summary>
    void OnValidate()
    {
        if (targetPlane != null && Application.isPlaying == false)
        {
            UpdateZoneFromPlane();
        }
    }
    
    /// <summary>
    /// Vérifie si une position est dans la zone de patrouille.
    /// </summary>
    public bool IsInZone(Vector3 position)
    {
        return position.x >= MinX && position.x <= MaxX &&
               position.z >= MinZ && position.z <= MaxZ;
    }
    
    /// <summary>
    /// ⚠️ MÉTHODE DÉPRÉCIÉE : Les clamps sont désactivés dans tout le projet.
    /// Cette méthode est conservée pour compatibilité mais ne fait plus de clamp.
    /// </summary>
    [System.Obsolete("Clamps are disabled. This method returns the position unchanged.")]
    public Vector3 ClampToZone(Vector3 position)
    {
        // ⚠️ SUPPRESSION DES CLAMPS : Retourner la position sans modification
        return position;
    }
    
    /// <summary>
    /// Génère une position aléatoire dans la zone.
    /// </summary>
    public Vector3 GetRandomPosition()
    {
        return new Vector3(
            Random.Range(MinX, MaxX),
            zoneY,
            Random.Range(MinZ, MaxZ)
        );
    }
    
    /// <summary>
    /// Génère une position aléatoire à l'extérieur de la zone de patrouille.
    /// Utile pour positionner l'intrus avant qu'il n'entre dans la zone.
    /// </summary>
    public Vector3 GetRandomPositionOutsideZone(float margin = 10f)
    {
        // Générer une position dans un anneau autour de la zone
        float angle = Random.Range(0f, 360f) * Mathf.Deg2Rad;
        float distance = Mathf.Max(zoneSizeX, zoneSizeZ) + margin + Random.Range(5f, 15f);
        
        Vector3 center = new Vector3(0, zoneY, 0);
        Vector3 position = center + new Vector3(
            Mathf.Cos(angle) * distance,
            0,
            Mathf.Sin(angle) * distance
        );
        
        return position;
    }
    
    /// <summary>
    /// Dessine la zone dans l'éditeur Unity (Gizmos).
    /// Pour voir les Gizmos : Sélectionner le GameObject avec PatrolZone dans la scène,
    /// ou cocher "Gizmos" en haut à droite de la fenêtre Scene.
    /// </summary>
    void OnDrawGizmos()
    {
        if (!showGizmos) return;
        
        // Mettre à jour depuis le plane si assigné
        if (targetPlane != null && Application.isPlaying == false)
        {
            UpdateZoneFromPlane();
        }
        
        Gizmos.color = gizmoColor;
        Vector3 center = new Vector3(0, zoneY, 0);
        Vector3 size = new Vector3(zoneSizeX * 2, 0.1f, zoneSizeZ * 2);
        Gizmos.DrawCube(center, size);
        
        // Dessiner le contour
        Gizmos.color = Color.green;
        Vector3 corner1 = new Vector3(MinX, zoneY, MinZ);
        Vector3 corner2 = new Vector3(MaxX, zoneY, MinZ);
        Vector3 corner3 = new Vector3(MaxX, zoneY, MaxZ);
        Vector3 corner4 = new Vector3(MinX, zoneY, MaxZ);
        
        Gizmos.DrawLine(corner1, corner2);
        Gizmos.DrawLine(corner2, corner3);
        Gizmos.DrawLine(corner3, corner4);
        Gizmos.DrawLine(corner4, corner1);
        
        // Dessiner les diagonales pour mieux voir
        Gizmos.color = Color.green * 0.5f;
        Gizmos.DrawLine(corner1, corner3);
        Gizmos.DrawLine(corner2, corner4);
    }
}

