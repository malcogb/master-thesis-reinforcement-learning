using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneAgent : MonoBehaviour
{
    // Vitesse de déplacement du drone
    public float moveSpeed = 7.5f;  // Augmenté de 5.0 à 7.5 (x1.5) pour améliorer la poursuite de l'intrus

    // Zone de patrouille (utilisée pour les limites)
    private PatrolZone patrolZone;

    void Start()
    {
        // Récupérer la zone de patrouille
        patrolZone = PatrolZone.Instance;
        if (patrolZone == null)
        {
            Debug.LogWarning("[DroneAgent] No PatrolZone found. Using default boundaries (-50 to 50).");
        }
    }

    void Update()
    {
        // Déplacement manuel (pour test uniquement)
        //float moveX = Input.GetAxis("Horizontal");
        //float moveZ = Input.GetAxis("Vertical");

        //Vector3 move = new Vector3(moveX, 0, moveZ) * moveSpeed * Time.deltaTime;
        //transform.position += move;

        // ⚠️ SUPPRESSION DES CLAMPS : Les drones peuvent se déplacer librement dans tout l'espace
    }
}
