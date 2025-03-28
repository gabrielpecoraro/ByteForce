# ByteForce Datascience Project

## Prérequis
Avant de commencer, assurez-vous d'avoir installé sur votre système :
- Git
- Un compte GitHub
- Python et pip

## Étape 1 : Cloner le repository
1. Ouvrez un terminal et exécutez la commande suivante :
   ```sh
   git clone <repository-url>
   ```
   Remplacez `<repository-url>` par l'URL du repository.
2. Accédez au dossier cloné :
   ```sh
   cd <repository-name>
   ```

## Étape 2 : Télécharger le dataset et le faiss_index
1. Téléchargez les fichiers nécessaires (dataset et faiss_index). https://drive.google.com/drive/folders/1792ircTT0FdIOWBJm6G1uWuM0vhI0toA
2. Placez-les directement dans le dossier du repository cloné, sans créer de sous-dossier, sauf si vous prévoyez de modifier le chemin d'accès dans le code.

## Étape 3 : Configurer l'environnement Python (si nécessaire)
1. (Optionnel) Créez un environnement virtuel :
   ```sh
   python -m venv env
   ```
2. Activez l'environnement virtuel :
   - Sur Windows :
     ```sh
     env\Scripts\activate
     ```
   - Sur macOS/Linux :
     ```sh
     source env/bin/activate
     ```

## Étape 4 : Installer les dépendances
1. Installez les packages nécessaires avec :
   ```sh
   pip install -r requirements.txt
   ```

## Étape 5 : Exécuter l'interface
1. Lancez l'application avec Streamlit :
   ```sh
   streamlit run INTERFACE/interface.py
   ```

