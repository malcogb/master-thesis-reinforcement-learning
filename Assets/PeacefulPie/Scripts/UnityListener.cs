using UnityEngine;

namespace PeacefulPie
{
    public class UnityListener : MonoBehaviour
    {
        public int port = 9000;

        void Awake()
        {
            var comms = gameObject.AddComponent<UnityComms>();
            comms.Port = port;
            Debug.Log($"[PeacefulPie] UnityListener Awake, will start UnityComms on port {port}");
        }
    }
}
