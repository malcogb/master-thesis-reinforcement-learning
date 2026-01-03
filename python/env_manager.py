import socket
import json
import time
import numpy as np

# Importer la configuration pour le délai entre steps
try:
    from config import STEP_DELAY, UNITY_RECONNECT_ENABLED, UNITY_RECONNECT_MAX_ATTEMPTS, UNITY_RECONNECT_DELAY, UNITY_SOCKET_TIMEOUT
except ImportError:
    STEP_DELAY = 0.0  # Pas de délai par défaut
    UNITY_RECONNECT_ENABLED = True
    UNITY_RECONNECT_MAX_ATTEMPTS = 5
    UNITY_RECONNECT_DELAY = 2.0
    UNITY_SOCKET_TIMEOUT = 15  # Timeout par défaut (15s)


class UnityEnvManager:
    def __init__(self, host='127.0.0.1', port=9000, step_delay=None):
        import sys
        print(f"🔍 [DIAGNOSTIC] UnityEnvManager.__init__ : Début (host={host}, port={port})...")
        sys.stdout.flush()
        
        self.host = host
        self.port = port
        self.sock = None
        
        print(f"🔍 [DIAGNOSTIC] UnityEnvManager.__init__ : Création du socket...")
        sys.stdout.flush()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(UNITY_SOCKET_TIMEOUT)  # Timeout configurable (15s par défaut pour détecter rapidement les blocages)
        
        # Tentative de connexion avec retry
        max_connect_attempts = 3
        connect_delay = 0.2
        print(f"🔍 [DIAGNOSTIC] UnityEnvManager.__init__ : Tentative de connexion à {host}:{port}...")
        sys.stdout.flush()
        
        for attempt in range(max_connect_attempts):
            try:
                print(f"🔍 [DIAGNOSTIC] UnityEnvManager.__init__ : Tentative {attempt + 1}/{max_connect_attempts}...")
                sys.stdout.flush()
                self.sock.connect((host, port))
                print(f"🔍 [DIAGNOSTIC] UnityEnvManager.__init__ : Connexion réussie !")
                sys.stdout.flush()
                break  # Connexion réussie
            except (ConnectionRefusedError, OSError) as e:
                print(f"⚠️  [DIAGNOSTIC] UnityEnvManager.__init__ : Tentative {attempt + 1} échouée : {e}")
                sys.stdout.flush()
                if attempt < max_connect_attempts - 1:
                    time.sleep(connect_delay)
                    self.sock.close()
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.settimeout(UNITY_SOCKET_TIMEOUT)  # Timeout configurable
                else:
                    print(f"❌ [DIAGNOSTIC] UnityEnvManager.__init__ : Toutes les tentatives ont échoué")
                    sys.stdout.flush()
                    raise ConnectionError(f"Impossible de se connecter à Unity après {max_connect_attempts} tentatives: {e}")
        
        # Délai entre chaque step (pour ralentir l'exécution si nécessaire)
        self.step_delay = step_delay if step_delay is not None else STEP_DELAY
        self.max_reconnect_attempts = UNITY_RECONNECT_MAX_ATTEMPTS
        self.reconnect_delay = UNITY_RECONNECT_DELAY

    def _reconnect(self):
        """Tente de se reconnecter à Unity."""
        if not UNITY_RECONNECT_ENABLED:
            raise ConnectionError("Reconnexion désactivée")
        
        # print(f"[Python] Tentative de reconnexion à Unity ({self.host}:{self.port})...")  # Réduire verbosité pour SubprocVecEnv
        for attempt in range(self.max_reconnect_attempts):
            try:
                if self.sock:
                    self.sock.close()
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(UNITY_SOCKET_TIMEOUT)  # Timeout configurable
                self.sock.connect((self.host, self.port))
                # print(f"[Python] ✅ Reconnexion réussie (tentative {attempt + 1}/{self.max_reconnect_attempts})")  # Réduire verbosité pour SubprocVecEnv
                return True
            except (ConnectionRefusedError, OSError) as e:
                if attempt < self.max_reconnect_attempts - 1:
                    # print(f"[Python] ⚠️  Tentative {attempt + 1}/{self.max_reconnect_attempts} échouée, nouvelle tentative dans {self.reconnect_delay}s...")  # Réduire verbosité
                    time.sleep(self.reconnect_delay)
                else:
                    print(f"[Python] ❌ Impossible de se reconnecter après {self.max_reconnect_attempts} tentatives")
                    raise ConnectionError(f"Reconnexion échouée: {e}")
        return False

    def send_message(self, msg_dict):
        msg = json.dumps(msg_dict)
        # print(f"[Python] Envoi : {msg}")  # Désactivé pour réduire la verbosité (SubprocVecEnv)
        try:
            self.sock.sendall(msg.encode('utf-8'))
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
            if UNITY_RECONNECT_ENABLED:
                # print(f"[Python] ⚠️  Erreur de connexion lors de l'envoi: {e}")  # Réduire verbosité pour SubprocVecEnv
                self._reconnect()
                # Réessayer l'envoi après reconnexion
                self.sock.sendall(msg.encode('utf-8'))
            else:
                raise

    def receive_message(self):
        data = b""
        max_retries = 3
        for retry in range(max_retries):
            try:
                while True:
                    try:
                        part = self.sock.recv(8192)  # Augmenté de 4096 à 8192 pour recevoir plus de données
                        if not part: 
                            if UNITY_RECONNECT_ENABLED and retry < max_retries - 1:
                                self._reconnect()
                                continue
                            break
                        data += part
                        # Vérifier si on a reçu un JSON complet (vérification améliorée)
                        if b"}" in part:
                            # Vérifier que les accolades sont équilibrées
                            text = data.decode('utf-8', errors='ignore')
                            if text.count('{') == text.count('}'):
                                break
                    except socket.timeout:
                        # Avec SubprocVecEnv, les timeouts peuvent être plus fréquents
                        if retry < max_retries - 1:
                            time.sleep(0.1)
                            continue
                        if UNITY_RECONNECT_ENABLED:
                            self._reconnect()
                            continue
                        break
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
                        if UNITY_RECONNECT_ENABLED and retry < max_retries - 1:
                            self._reconnect()
                            continue
                        raise
                
                if len(data) == 0:
                    if retry < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    raise ConnectionError("Aucune donnée reçue de Unity")
                
                text = data.decode('utf-8')
                return json.loads(text)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
                if UNITY_RECONNECT_ENABLED and retry < max_retries - 1:
                    self._reconnect()
                    time.sleep(0.5)
                    continue
                raise
            except json.JSONDecodeError as e:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                    data = b""
                    continue
                raise
        
        raise ConnectionError("Impossible de recevoir un message valide de Unity")

    def reset(self, stage=None, intruder_speed_mult=None, enable_obstacles=None, enable_zone=None):
        """
        Réinitialise l'environnement Unity.
        
        Args:
            stage: Stage actuel du curriculum (0, 1, ou 2)
            intruder_speed_mult: Multiplicateur de vitesse de l'intrus
            enable_obstacles: Activer/désactiver les obstacles
            enable_zone: Activer/désactiver la zone de patrouille
        
        Note: Les drones sont créés UNIQUEMENT au démarrage Unity, pas de mise à jour pendant l'entraînement.
        """
        import sys
        print(f"🔍 [DIAGNOSTIC] UnityEnvManager.reset : Début (stage={stage})...")
        sys.stdout.flush()
        
        reset_msg = {"command": "reset"}
        
        # 🎓 Curriculum Learning: Ajouter les paramètres du stage si fournis
        if stage is not None:
            reset_msg["stage"] = stage
        if intruder_speed_mult is not None:
            reset_msg["intruder_speed_mult"] = intruder_speed_mult
        if enable_obstacles is not None:
            reset_msg["enable_obstacles"] = enable_obstacles
        if enable_zone is not None:
            reset_msg["enable_zone"] = enable_zone
        
        print(f"🔍 [DIAGNOSTIC] UnityEnvManager.reset : Envoi du message reset...")
        sys.stdout.flush()
        try:
            self.send_message(reset_msg)
            print(f"🔍 [DIAGNOSTIC] UnityEnvManager.reset : Message envoyé, attente de la réponse...")
            sys.stdout.flush()
            response = self.receive_message()
            print(f"🔍 [DIAGNOSTIC] UnityEnvManager.reset : Réponse reçue")
            sys.stdout.flush()
            time.sleep(0.05)  # 🔹 très léger délai pour Unity
            return response
        except Exception as e:
            print(f"❌ [DIAGNOSTIC] UnityEnvManager.reset : Erreur : {e}")
            sys.stdout.flush()
            raise

    def step(self, actions):
        import sys
        # Conversion numpy -> liste native
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        
        try:
            self.send_message({"actions": actions})
            response = self.receive_message()
            
            # Ajouter un délai entre les steps pour ralentir l'exécution (si configuré)
            if self.step_delay > 0:
                time.sleep(self.step_delay)
            
            return response
        except Exception as e:
            print(f"❌ [DIAGNOSTIC] UnityEnvManager.step : Erreur : {e}")
            sys.stdout.flush()
            raise

    def close(self):
        """Ferme proprement la connexion Unity."""
        try:
            if hasattr(self, "proc") and self.proc:
                print("[Python] Fermeture du processus Unity...")
                self.proc.terminate()
                self.proc.wait()
            if hasattr(self, "sock") and self.sock:
                self.sock.close()
            print("[Python] Connexion Unity fermée proprement.")
        except Exception as e:
            print(f"[Python] Erreur lors de la fermeture : {e}")


