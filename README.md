# Agent conversationnel du département informatique de l'université Lyon 1

Cet agent conversationnel a pour objectif d'informer toutes personnes voulant intégrer une formation du département informatique de l'université Claude Bernard. Le Pr Alexandre Aussem est à l'initiative de ce projet en le proposant à ses étudiants comme stage.

Le projet couvre pour l'instant qu'une formation : le master Data Science.

## Agent conversationnel Data Science

### Guide d'installation

Avant d'installer _ChatbotDS_, il est recommandé d'utiliser un environnement virtuel.

```bash
python -m venv venv
```

Il faut ensuite activer l'environnement virtuel suivant votre os.

Linux or MacOS :

```bash
source ./venv/bin/activate
```

Windows (CMD) :

```cmd
.\venv\Scripts\activate.bat
```

Windows (PowerShell) :

```cmd
.\venv\Scripts\Activate.ps1
```

On peut maintenant installer _ChatbotDS_.

```bash
pip install -e ./ChatbotDS
```

## Publication

**Poster**: _Florian Baud_ et _Alex Aussem_, Répondre aux requêtes des étudiants avec un agent conversationnel à mémoire supervisée, EGC Janvier 2023, Lyon

**Papier**: Baud, F., & Aussem, A. (2023). Answering Student Queries with a Supervised Memory Conversational Agent. _The International FLAIRS Conference Proceedings_, 36(1). [https://doi.org/10.32473/flairs.36.133195](https://doi.org/10.32473/flairs.36.133195)
