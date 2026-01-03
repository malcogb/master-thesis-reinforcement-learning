"""
Fonctions utilitaires pour le projet.
"""
import os

def ensure_dir(directory):
    """
    Crée un répertoire s'il n'existe pas.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Directory created: {directory}")
