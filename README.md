# Agent conversationnel du département informatique de l'université Lyon 1

Cet agent conversationnel a pour objectif d'informer toutes personnes voulant intégrer une formation du département informatique de l'université Claude Bernard. Le Pr Alexandre Aussem est à l'initiative de ce projet en le proposant à ses étudiants comme stage.

Le projet couvre pour l'instant qu'une formation : le master Data Science.

## Agent conversationnel Data Science

**Guide d'installation**

Avant d'installer _ChatbotDS_, il est recommandé d'utiliser un environnement virtuel.

```bash
$ python -m venv venv
```

Il faut ensuite activer l'environnement virtuel suivant votre os.

Linux or MacOS :
```bash
$ source ./venv/bin/activate
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
$ pip install -e ./ChatbotDS
```

**Procédure de deploiement :**

```bash
$ sudo docker-compose down
$ sudo git pull
$ sudo docker-compose build
$ sudo docker-compose up -d
```