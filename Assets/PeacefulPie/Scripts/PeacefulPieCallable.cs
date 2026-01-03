using UnityEngine;

namespace PeacefulPie
{
    [DisallowMultipleComponent]
    public class PeacefulPieCallable : MonoBehaviour
    {
        [Tooltip("Nom de l'objet callable (identifiant pour les appels RPC).")]
        public string callableName;

        void Awake()
        {
            if (string.IsNullOrEmpty(callableName))
                callableName = gameObject.name;

            Debug.Log($"[PeacefulPieCallable] Registered callable object: {callableName}");
        }

        public void CallFromPython(string method, string args)
        {
            Debug.Log($"[PeacefulPieCallable] Python called {method}({args}) on {callableName}");
            // Ici tu pourras plus tard exécuter des actions Unity selon le message reçu
        }
    }
}
