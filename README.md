# 🧠 Offline to Online TD3-BC

A repository for offline to online RL using the TD3-BC algorithm.

---

## 🚀 First-Time Setup

Follow these steps to set up your development environment:

1. **Create a virtual environment with [`uv`](https://github.com/astral-sh/uv):**

   ```bash
   uv venv
   ```

2. **Install all dependencies:**

   ```bash
   uv sync
   ```

---

## 🛠️ Usage Guide

### ✅ Activate the Virtual Environment

Before running anything, activate the environment:

```bash
source .venv/bin/activate
```

### 🏋️‍♀️ Start Training

Launch training using one of the available configuration files:

```bash
python main.py --config_path config/config_pretrain.yaml
```

#### 📁 Available Configurations

* `config/config_pretrain.yaml` — Pretraining phase (offline)
* `config/config_refine.yaml` — Refinement phase (fine-tuning)
* `config/config_online.yaml` — Online learning phase

---

## 📌 Notes

* Ensure [`uv`](https://github.com/astral-sh/uv) is installed before proceeding.