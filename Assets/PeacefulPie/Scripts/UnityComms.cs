using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json.Linq;
using System.Collections.Concurrent;

namespace PeacefulPie
{
    public class UnityComms : MonoBehaviour
    {
        public int Port = 9000;

        [Header("Agents assignés via EnvManager")]
        public List<GameObject> drones;
        public GameObject intruder;
        public List<GameObject> obstacles;
        
        [Header("Movement Settings")]
        public float droneMoveSpeed = 1.0f;  // Vitesse de déplacement des drones (multiplie les actions)
        
        [Header("Collision Settings")]
        [Tooltip("Rayon de détection de collision pour les drones (distance minimale aux obstacles)")]
        public float droneCollisionRadius = 0.5f;  // Rayon de collision des drones
        
        [Tooltip("Si activé, les drones ne peuvent pas traverser les obstacles")]
        public bool preventObstacleCollision = true;  // Empêcher les collisions avec les obstacles
        
        private PatrolZone patrolZone;  // Zone de patrouille (utilisée pour les calculs, pas pour le clamp)
        private GameObject planeObject;  // 🎓 Plane GameObject pour gérer sa visibilité selon le stage
        private GameObject obstaclesParentObject;  // 🎓 GameObject parent "Obstacles" (Empty)
        private GameObject patrolZoneObject;  // 🎓 GameObject PatrolZone (Empty)

        private TcpListener listener;
        private Thread listenerThread;
        private bool isRunning = false;
        private int stepCounter = 0; // ✅ compteur de steps IA
        
        // Curriculum Learning - Stage actuel
        private int currentStage = 0;  // Stage actuel (0, 1, ou 2)
        private bool enableObstacles = false;  // Obstacles activés selon le stage
        private bool enableZone = false;  // Zone activée selon le stage

        private ConcurrentQueue<ClientRequest> requestQueue = new ConcurrentQueue<ClientRequest>();

        private class ClientRequest
        {
            public JObject command;
            public NetworkStream stream;
        }

        void Start()
        {
            // Récupérer la zone de patrouille
            patrolZone = PatrolZone.Instance;
            
            // 🎓 Récupérer les GameObjects Empty pour gérer leur visibilité
            // 1. Plane GameObject
            if (patrolZone != null && patrolZone.targetPlane != null)
            {
                planeObject = patrolZone.targetPlane;
            }
            else
            {
                // Fallback : chercher directement dans la scène
                planeObject = GameObject.Find("Plane");
            }
            
            // 2. Obstacles parent GameObject (Empty)
            // Essayer plusieurs méthodes pour trouver le GameObject "Obstacles"
            obstaclesParentObject = GameObject.Find("Obstacles");
            
            // 3. PatrolZone GameObject (Empty - celui avec le script PatrolZone)
            if (patrolZone != null)
            {
                patrolZoneObject = patrolZone.gameObject;
            }
            
            // 🔹 Récupérer les drones, intrus et obstacles depuis EnvManager
            // Les drones sont maintenant créés au démarrage dans EnvManager.Start()
            var env = FindObjectOfType<EnvManager>();
            
            // Si GameObject.Find ne trouve pas le parent "Obstacles", essayer via EnvManager qui a les obstacles assignés
            if (obstaclesParentObject == null && env != null)
            {
                if (env.obstaclesStatic != null && env.obstaclesStatic.Count > 0)
                {
                    // Prendre le parent du premier obstacle
                    if (env.obstaclesStatic[0] != null && env.obstaclesStatic[0].transform.parent != null)
                    {
                        obstaclesParentObject = env.obstaclesStatic[0].transform.parent.gameObject;
                        Debug.Log($"[UnityComms] Obstacles parent GameObject found via EnvManager (first obstacle parent): {obstaclesParentObject.name}");
                    }
                }
            }
            
            if (obstaclesParentObject == null)
            {
                Debug.LogWarning("[UnityComms] Obstacles parent GameObject not found with GameObject.Find('Obstacles'). Will try to find it later when obstacles are assigned via AssignObstacles().");
            }
            else
            {
                Debug.Log($"[UnityComms] Obstacles parent GameObject found: {obstaclesParentObject.name}");
            }
            if (env != null)
            {
                // Attendre un frame pour que les drones soient créés dans EnvManager.Start()
                // Les drones seront assignés automatiquement par EnvManager après leur création
                // On vérifie quand même ici au cas où
                if (env.dronesStatic != null && env.dronesStatic.Count > 0)
                    AssignDrones(env.dronesStatic);
                if (env.intruderStatic != null)
                    AssignIntruder(env.intruderStatic);
                if (env.obstaclesStatic != null && env.obstaclesStatic.Count > 0)
                {
                    AssignObstacles(env.obstaclesStatic);
                    // 🔹 Si obstaclesParentObject n'a pas été trouvé, essayer de le trouver via les obstacles assignés
                    if (obstaclesParentObject == null && obstacles != null && obstacles.Count > 0)
                    {
                        if (obstacles[0] != null && obstacles[0].transform.parent != null)
                        {
                            obstaclesParentObject = obstacles[0].transform.parent.gameObject;
                            Debug.Log($"[UnityComms] Obstacles parent GameObject found via AssignObstacles: {obstaclesParentObject.name}");
                        }
                    }
                }
            }
            
            // 🎓 Stage 0 : Désactiver tous les GameObjects Empty (Plane, Obstacles, PatrolZone) au démarrage
            SetPlaneVisibility(false);
            SetObstaclesEnabled(false);
            SetObstaclesParentVisibility(false);
            SetPatrolZoneVisibility(false);
            
            // 🚁 Initialiser la liste des drones (vide pour l'instant)
            if (drones == null)
            {
                drones = new List<GameObject>();
            }

            StartServer();
        }

        void OnApplicationQuit()
        {
            StopServer();
        }

        void Update()
        {
            // 🔹 Traiter les requêtes dans le thread principal Unity
            while (requestQueue.TryDequeue(out ClientRequest req))
            {
                ApplyActionsFromJson(req.command);

                string response = GetAgentsStateJson(req.command);
                byte[] responseBytes = Encoding.UTF8.GetBytes(response);

                try
                {
                    req.stream.Write(responseBytes, 0, responseBytes.Length);
                    req.stream.Flush(); // ✅ assure l’envoi complet
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[UnityComms] Failed to send response: {e}");
                }
            }
        }

        #region Server Control
        public void StartServer()
        {
            if (isRunning) return;

            isRunning = true;
            listenerThread = new Thread(ListenLoop) { IsBackground = true };
            listenerThread.Start();
            Debug.Log($"[UnityComms] Server started on port {Port}");
        }

        public void StopServer()
        {
            isRunning = false;
            listener?.Stop();
            if (listenerThread != null && listenerThread.IsAlive)
                listenerThread.Abort();
            listenerThread = null;
            Debug.Log("[UnityComms] Server stopped.");
        }
        #endregion

        #region Networking
        private void ListenLoop()
        {
            try
            {
                listener = new TcpListener(IPAddress.Any, Port);
                listener.Server.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                listener.Start();
                Debug.Log($"[UnityComms] Listening on port {Port}");

                while (isRunning)
                {
                    if (listener.Pending())
                    {
                        TcpClient client = listener.AcceptTcpClient();
                        ThreadPool.QueueUserWorkItem(HandleClient, client);
                    }
                    else
                    {
                        Thread.Sleep(50);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[UnityComms] Exception: {e}");
            }
            finally
            {
                listener?.Stop();
            }
        }

        private void HandleClient(object clientObj)
        {
            TcpClient client = (TcpClient)clientObj;
            NetworkStream stream = client.GetStream();
            stream.ReadTimeout = 10000; // 10 secondes max

            try
            {
                byte[] buffer = new byte[4096];

                while (isRunning && client.Connected)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead == 0) break; // client fermé

                    string message = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    JObject jObj = null;

                    try { jObj = JObject.Parse(message); }
                    catch { Debug.LogWarning("[UnityComms] JSON vide ou malformé."); }

                    if (jObj != null)
                        requestQueue.Enqueue(new ClientRequest { command = jObj, stream = stream });
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[UnityComms] Client disconnected or error: {e.Message}");
            }
            finally
            {
                stream.Close();
                client.Close();
            }
        }

        #endregion

        #region Action Handling
        private void ApplyActionsFromJson(JObject jObj)
        {
            if (jObj["command"] != null && jObj["command"].ToString() == "reset")
            {
                // Récupérer EnvManager une seule fois pour toute la méthode
                var env = FindObjectOfType<EnvManager>();
                if (env == null)
                {
                    Debug.LogError("[UnityComms] EnvManager not found in scene!");
                    return;
                }
                
                // 🚁 Les drones sont maintenant créés au démarrage (Play) dans EnvManager.Start()
                // Vérifier que les drones existent et les assigner si nécessaire
                if (drones == null || drones.Count == 0)
                {
                    if (env.dronesStatic != null && env.dronesStatic.Count > 0)
                    {
                        AssignDrones(env.dronesStatic);
                        Debug.Log($"[UnityComms] ✅ {env.dronesStatic.Count} drones assignés depuis EnvManager.");
                    }
                    else
                    {
                        Debug.LogWarning("[UnityComms] Aucun drone trouvé dans EnvManager. Les drones devraient être créés au démarrage.");
                    }
                }
                
                // 🎓 Curriculum Learning: Récupérer les paramètres du stage depuis Python
                if (jObj["stage"] != null)
                {
                    currentStage = jObj["stage"].ToObject<int>();
                    Debug.Log($"[UnityComms] Curriculum Stage {currentStage} received from Python.");
                }
                
                if (jObj["intruder_speed_mult"] != null)
                {
                    float speedMult = jObj["intruder_speed_mult"].ToObject<float>();
                    // Appliquer le multiplicateur de vitesse à l'intrus
                    var intruderAgent = intruder?.GetComponent<IntruderAgent>();
                    if (intruderAgent != null)
                    {
                        intruderAgent.SetSpeedMultiplier(speedMult);
                        Debug.Log($"[UnityComms] Intruder speed multiplier set to {speedMult}x");
                    }
                }
                
                if (jObj["enable_obstacles"] != null)
                {
                    enableObstacles = jObj["enable_obstacles"].ToObject<bool>();
                    
                    // 🔹 Réassigner les obstacles depuis EnvManager au cas où ils n'ont pas été assignés au démarrage
                    if ((obstacles == null || obstacles.Count == 0) && env != null)
                    {
                        if (env.obstaclesStatic != null && env.obstaclesStatic.Count > 0)
                        {
                            AssignObstacles(env.obstaclesStatic);
                            Debug.Log($"[UnityComms] Obstacles reassigned from EnvManager: {obstacles.Count} obstacles");
                        }
                        else
                        {
                            Debug.LogWarning($"[UnityComms] EnvManager.obstaclesStatic is null or empty! Cannot enable obstacles. Please assign obstacles to EnvManager.obstaclesStatic in Unity Inspector.");
                        }
                    }
                    
                    // Activer/désactiver les obstacles selon le stage
                    SetObstaclesEnabled(enableObstacles);
                    // 🎓 Gérer aussi la visibilité du GameObject parent "Obstacles"
                    SetObstaclesParentVisibility(enableObstacles);
                    Debug.Log($"[UnityComms] Obstacles {(enableObstacles ? "enabled" : "disabled")} for stage {currentStage}");
                }
                
                if (jObj["enable_zone"] != null)
                {
                    enableZone = jObj["enable_zone"].ToObject<bool>();
                    // 🎓 Gérer la visibilité du Plane selon le stage
                    SetPlaneVisibility(enableZone);
                    // 🎓 Gérer aussi la visibilité du GameObject PatrolZone
                    SetPatrolZoneVisibility(enableZone);
                    Debug.Log($"[UnityComms] Zone {(enableZone ? "enabled" : "disabled")} for stage {currentStage}");
                }
                
                env.ResetEnv();
                stepCounter = 0; // 🔄 réinitialisation du compteur
                Debug.Log($"[UnityComms] Reset command applied. StepCounter reset. Stage: {currentStage}");
                return;
            }

            if (drones == null || drones.Count == 0)
            {
                Debug.LogWarning("[UnityComms] No drones assigned. Cannot apply actions.");
                return;
            }

            JArray actions = jObj["actions"] as JArray;
            if (actions == null || actions.Count == 0)
            {
                Debug.LogWarning("[UnityComms] No actions received or invalid format.");
                return;
            }
            
            if (actions.Count != drones.Count)
            {
                Debug.LogWarning($"[UnityComms] Action count ({actions.Count}) doesn't match drone count ({drones.Count}).");
            }

            for (int i = 0; i < drones.Count && i < actions.Count; i++)
            {
                if (drones[i] == null)
                {
                    Debug.LogWarning($"[UnityComms] Drone {i} is null, skipping action.");
                    continue;
                }
                
                float dx = actions[i][0].ToObject<float>();
                float dz = actions[i][1].ToObject<float>();
                
                // Appliquer la vitesse de déplacement (les actions sont entre -1 et 1)
                Vector3 movement = new Vector3(dx * droneMoveSpeed, 0, dz * droneMoveSpeed);
                Vector3 newPosition = drones[i].transform.position + movement;
                
                // 🎓 Curriculum: Vérifier les collisions avec les obstacles seulement si activés
                if (enableObstacles && preventObstacleCollision && obstacles != null && obstacles.Count > 0)
                {
                    newPosition = CheckObstacleCollision(drones[i].transform.position, newPosition, i);
                }
                
                // Appliquer le mouvement (après vérification des collisions)
                drones[i].transform.position = newPosition;
                
                // ⚠️ SUPPRESSION DES CLAMPS : Les drones peuvent se déplacer librement dans tout l'espace
            }

            stepCounter++; // ✅ incrément par action
            Debug.Log($"[UnityComms] Actions applied to {drones.Count} drones. Step {stepCounter}");
        }
        /// <summary>
        /// Vérifie si le mouvement proposé causerait une collision avec un obstacle.
        /// Retourne la position corrigée (sans collision) ou la position originale si collision détectée.
        /// </summary>
        private Vector3 CheckObstacleCollision(Vector3 currentPos, Vector3 newPos, int droneIndex)
        {
            if (obstacles == null || obstacles.Count == 0)
                return newPos;
            
            // Vérifier chaque obstacle
            foreach (var obstacle in obstacles)
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
                
                // Distance minimale requise (rayon obstacle + rayon drone + petite marge)
                float minDistance = obstacleRadius + droneCollisionRadius + 0.1f;
                
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
                    
                    // Si le drone est exactement sur l'obstacle (distance = 0), utiliser une direction par défaut
                    if (directionFromObstacle.magnitude < 0.01f)
                    {
                        directionFromObstacle = Vector3.right; // Direction par défaut
                    }
                    
                    // Position sécurisée : à la distance minimale de l'obstacle, dans la direction opposée
                    float safeDistance = minDistance - 0.05f; // Petite marge de sécurité
                    Vector3 safePosition = obstaclePos2D + directionFromObstacle * safeDistance;
                    
                    // Garder la hauteur Y originale
                    safePosition.y = newPos.y;
                    
                    // ⚠️ SUPPRESSION DES CLAMPS : Retourner la position sécurisée sans vérification de zone
                    return safePosition;
                }
            }
            
            // Aucune collision détectée, mouvement autorisé
            return newPos;
        }
        #endregion

        #region State Reporting
        private string GetAgentsStateJson(JObject lastCommand = null)
        {
            JObject state = new JObject();
            JArray dronesArray = new JArray();

            if (drones != null)
            {
                foreach (var d in drones)
                {
                    if (d == null) continue;  // Ignorer les drones null
                    
                    Vector3 pos = d.transform.position;
                    dronesArray.Add(new JObject
                    {
                        ["x"] = pos.x,
                        ["y"] = pos.y,
                        ["z"] = pos.z
                    });
                }
            }

            state["drones"] = dronesArray;

            if (intruder != null)
            {
                Vector3 pos = intruder.transform.position;
                state["intruder"] = new JObject
                {
                    ["x"] = pos.x,
                    ["y"] = pos.y,
                    ["z"] = pos.z
                };
            }
            else
            {
                state["intruder"] = null;
            }

            // Ajouter les positions des obstacles
            JArray obstaclesArray = new JArray();
            if (obstacles != null)
            {
                foreach (var obs in obstacles)
                {
                    if (obs == null) continue;
                    
                    Vector3 pos = obs.transform.position;
                    Vector3 scale = obs.transform.localScale;
                    
                    // Obtenir les dimensions du collider si disponible
                    float radius = 1.0f;
                    Collider col = obs.GetComponent<Collider>();
                    if (col != null)
                    {
                        if (col is BoxCollider boxCol)
                        {
                            radius = Mathf.Max(boxCol.size.x, boxCol.size.z) * 0.5f;
                        }
                        else if (col is SphereCollider sphereCol)
                        {
                            radius = sphereCol.radius;
                        }
                        else if (col is CapsuleCollider capsuleCol)
                        {
                            radius = capsuleCol.radius;
                        }
                    }
                    
                    obstaclesArray.Add(new JObject
                    {
                        ["x"] = pos.x,
                        ["y"] = pos.y,
                        ["z"] = pos.z,
                        ["radius"] = radius * Mathf.Max(scale.x, scale.z)  // Rayon effectif de l'obstacle
                    });
                }
            }
            state["obstacles"] = obstaclesArray;

            state["step"] = stepCounter;
            state["timestamp"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ");
            state["status"] = lastCommand != null && lastCommand["command"]?.ToString() == "reset"
                ? "reset_done"
                : "received";

            return state.ToString();
        }
        #endregion

        #region Dynamic Assignment
        public void AssignDrones(List<GameObject> droneList)
        {
            drones = droneList;
            Debug.Log($"[UnityComms] {drones.Count} drones assigned.");
        }

        public void AssignIntruder(GameObject intruderObj)
        {
            intruder = intruderObj;
            Debug.Log($"[UnityComms] Intruder assigned: {intruder.name}");
        }
        
        public void AssignObstacles(List<GameObject> obstacleList)
        {
            obstacles = obstacleList;
            Debug.Log($"[UnityComms] {obstacles.Count} obstacles assigned.");
            
            // 🔹 Si obstaclesParentObject n'a pas été trouvé au Start(), essayer de le trouver maintenant
            if (obstaclesParentObject == null && obstacles != null && obstacles.Count > 0)
            {
                // Prendre le parent du premier obstacle
                if (obstacles[0] != null && obstacles[0].transform.parent != null)
                {
                    obstaclesParentObject = obstacles[0].transform.parent.gameObject;
                    Debug.Log($"[UnityComms] Obstacles parent GameObject found via first obstacle parent: {obstaclesParentObject.name}");
                }
                else
                {
                    // Essayer GameObject.Find une dernière fois
                    obstaclesParentObject = GameObject.Find("Obstacles");
                    if (obstaclesParentObject != null)
                    {
                        Debug.Log($"[UnityComms] Obstacles parent GameObject found with GameObject.Find: {obstaclesParentObject.name}");
                    }
                }
            }
        }
        
        /// <summary>
        /// 🎓 Curriculum Learning: Active ou désactive les obstacles selon le stage.
        /// Au Stage 0, les obstacles sont désactivés (SetActive(false)).
        /// </summary>
        private void SetObstaclesEnabled(bool enabled)
        {
            if (obstacles == null)
            {
                Debug.LogWarning($"[UnityComms] Cannot {(enabled ? "activate" : "deactivate")} obstacles: obstacles list is null!");
                return;
            }
            
            if (obstacles.Count == 0)
            {
                Debug.LogWarning($"[UnityComms] Cannot {(enabled ? "activate" : "deactivate")} obstacles: obstacles list is empty! Check if obstacles are assigned to EnvManager.obstaclesStatic.");
                return;
            }
            
            int activatedCount = 0;
            foreach (var obstacle in obstacles)
            {
                if (obstacle != null)
                {
                    obstacle.SetActive(enabled);
                    if (enabled && obstacle.activeInHierarchy)
                        activatedCount++;
                }
            }
            
            Debug.Log($"[UnityComms] Obstacles {(enabled ? "activated" : "deactivated")} (Stage {currentStage}) - {activatedCount}/{obstacles.Count} obstacles {(enabled ? "active" : "inactive")}");
        }
        
        /// <summary>
        /// 🎓 Curriculum Learning: Active ou désactive la visibilité du Plane selon le stage.
        /// Au Stage 0, le Plane est désactivé (SetActive(false)).
        /// </summary>
        private void SetPlaneVisibility(bool visible)
        {
            if (planeObject != null)
            {
                planeObject.SetActive(visible);
                Debug.Log($"[UnityComms] Plane {(visible ? "activated" : "deactivated")} (Stage {currentStage})");
            }
            else
            {
                Debug.LogWarning("[UnityComms] Plane GameObject not found. Cannot control visibility.");
            }
        }
        
        /// <summary>
        /// 🎓 Curriculum Learning: Active ou désactive la visibilité du GameObject parent "Obstacles" selon le stage.
        /// Au Stage 0, le GameObject "Obstacles" est désactivé (SetActive(false)).
        /// </summary>
        private void SetObstaclesParentVisibility(bool visible)
        {
            if (obstaclesParentObject != null)
            {
                obstaclesParentObject.SetActive(visible);
                Debug.Log($"[UnityComms] Obstacles parent GameObject {(visible ? "activated" : "deactivated")} (Stage {currentStage})");
            }
            else
            {
                Debug.LogWarning($"[UnityComms] Obstacles parent GameObject not found! Cannot {(visible ? "activate" : "deactivate")}. Make sure a GameObject named 'Obstacles' exists in the scene. Obstacles may not be visible even if individually activated.");
            }
        }
        
        /// <summary>
        /// 🎓 Curriculum Learning: Active ou désactive la visibilité du GameObject PatrolZone selon le stage.
        /// Au Stage 0, le GameObject PatrolZone est désactivé (SetActive(false)).
        /// </summary>
        private void SetPatrolZoneVisibility(bool visible)
        {
            if (patrolZoneObject != null)
            {
                patrolZoneObject.SetActive(visible);
                Debug.Log($"[UnityComms] PatrolZone GameObject {(visible ? "activated" : "deactivated")} (Stage {currentStage})");
            }
            else
            {
                Debug.LogWarning("[UnityComms] PatrolZone GameObject not found. Cannot control visibility.");
            }
        }
        #endregion
    }
}
