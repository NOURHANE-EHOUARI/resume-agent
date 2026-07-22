# Résumé — Agent de résumé automatique de texte

Application web permettant de générer automatiquement le résumé d'un document
(texte, PDF ou Word), via un modèle de langage exécuté **en local** sur la
machine, sans dépendre d'une API externe payante.

## Fonctionnement

1. L'utilisateur dépose un fichier (`.txt`, `.pdf` ou `.docx`) sur l'interface web
2. Le texte est extrait automatiquement du fichier
3. Le texte est envoyé au modèle de langage **Llama 3.2 (3B)**, exécuté
   localement via [Ollama](https://ollama.com)
4. Le résumé généré s'affiche directement dans l'interface

Le modèle est piloté par un prompt système conçu spécifiquement pour ce projet,
qui impose plusieurs règles :
- rester strictement fidèle au contenu du document (pas d'invention de faits,
  de dates, de lieux ou de détails absents du texte original)
- adapter la longueur du résumé au contenu, sans longueur imposée arbitrairement
- toujours terminer le résumé proprement, sans le couper en cours de génération
- traiter correctement les documents structurés (CV, fiches techniques,
  documents à sections) en évitant de les transformer en récit inventé

## Aperçu de l'interface

- Zone de dépôt de fichier par glisser-déposer ou sélection manuelle
- Génération du résumé en un clic, avec indicateur de chargement
- Résumé affiché dans une carte dédiée, avec bouton de copie rapide

## Stack technique

| Composant | Rôle |
|---|---|
| **Flask** | Serveur web / API backend |
| **Ollama** | Exécution locale du modèle Llama 3.2, sans cloud ni API payante |
| **pypdf** | Extraction du texte depuis les fichiers PDF |
| **python-docx** | Extraction du texte depuis les fichiers Word (.docx) |
| HTML / CSS / JavaScript | Interface utilisateur (sans framework front-end) |

## Installation

### Prérequis

- Python 3.12+
- [Ollama](https://ollama.com/download) installé sur la machine

### Étapes

```bash
# Cloner le repository
git clone https://github.com/NOURHANE-EHOUARI/resume-agent.git
cd resume-agent

# Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Installer les dépendances Python
pip install -r requirements.txt

# Télécharger le modèle Llama 3.2 (3B) via Ollama
ollama pull llama3.2:3b
```

### Lancer l'application

```bash
python app.py
```

Puis ouvrir **http://localhost:5000** dans le navigateur.

> Ollama doit être installé et actif en arrière-plan (il démarre automatiquement
> après installation) pour que la génération de résumé fonctionne.
