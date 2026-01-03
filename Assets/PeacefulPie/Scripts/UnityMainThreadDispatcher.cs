using System;
using System.Collections.Generic;
using UnityEngine;
using PeacefulPie;


namespace PeacefulPie
{
    public class UnityMainThreadDispatcher : MonoBehaviour
    {
        private static readonly Queue<Action> _executionQueue = new Queue<Action>();
        private static UnityMainThreadDispatcher _instance = null;

        void Awake()
        {
            // Singleton: s'assurer qu'une seule instance existe
            if (_instance == null)
            {
                _instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else if (_instance != this)
            {
                Destroy(gameObject);
            }
        }

        public static UnityMainThreadDispatcher Instance()
        {
            if (_instance == null)
            {
                // Chercher dans la scène existante
                _instance = FindObjectOfType<UnityMainThreadDispatcher>();
                if (_instance == null)
                {
                    Debug.LogError("UnityMainThreadDispatcher instance not found in scene! " +
                                   "Please add it to a GameObject manually.");
                }
            }
            return _instance;
        }

        void Update()
        {
            lock (_executionQueue)
            {
                while (_executionQueue.Count > 0)
                {
                    _executionQueue.Dequeue().Invoke();
                }
            }
        }

        public void Enqueue(Action action)
        {
            if (action == null) return;
            lock (_executionQueue)
            {
                _executionQueue.Enqueue(action);
            }
        }

        public static void EnqueueStatic(Action action)
        {
            var dispatcher = Instance();
            if (dispatcher != null)
                dispatcher.Enqueue(action);
        }
    }
}

