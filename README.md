# Streamlit App 

This project runs a **Streamlit** application using **Docker Compose** and **GNU Make** , with all commands run from **Git Bash** on Windows.

---

## Required Tools (Install via Git Bash on Windows)

| Tool                 | Description                      | Install Instructions                                         |
| -------------------- | -------------------------------- | ------------------------------------------------------------ |
| **Docker Desktop**   | Required for containers          | [Install Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) |
| **Git Bash**         | Terminal to run `make`on Windows | [Install Git for Windows](https://git-scm.com/download/win) _(includes Git Bash)_ |
| **Make for Windows** | Needed to run `make up`commands  | Install using [Gow](https://github.com/bmatzelle/gow) or follow steps below |

---

### 🛠 Install `make` via Git Bash (using Chocolatey)

If you don't have **make** installed, follow these steps:

1. **Install [Chocolatey](https://chocolatey.org/install)** (you’ll need **Admin PowerShell** ):

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; `
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```

2. **Install `make`** :

```bash
choco install make
```

3. **Restart Git Bash or your PC** to ensure `make` is in your `PATH`.

---

## How to Run the App (Using Git Bash + Make)

### 1. Open **Git Bash** and go to the project directory:

```bash
cd /c/path/to/your/rootfolder/ops
```

> Replace `/c/path/...` with your actual project location.

---

### 2. Build and start the app:

```bash
make up
```

This will:

- Build the Docker image
- Run the Streamlit app
- Open it at: [http://localhost:8501](http://localhost:8501/)

---

### 3. Stop the app:

```bash
make down
```

---

## Project Structure

```
├── app.py
├── data
│   ├── final_combined.parquet
│   ├── similarity_results_relaxed.csv
│   ├── similarity_results_strict.csv
│   └── symbols-info.parquet
├── disp_types.py
├── distance_engine.py
├── exp
│   └── test.sql
├── model
│   └── trading_model_final.keras
├── model.py
├── ops
│   ├── docker-compose.yml
│   ├── dockerfile
│   └── Makefile
├── README.md
├── requirements.txt
├── scratch.py
├── sequence_encoder.py
├── Trading_LSTM_Training.ipynb
└── utils.py
```

---

## What's in the Makefile?

```make
build:
	docker compose -f ./docker-compose.yml build

up: build
	docker compose -f ./docker-compose.yml up

down:
	docker compose -f ./docker-compose.yml down
```

---

## Tips

- If `make` is still not found in Git Bash, try `where make` or add its install path to the environment variables.
- Restart Docker Desktop before running if it's idle.
- Make sure your `requirements.txt` is in the root directory (not inside `/ops`).
