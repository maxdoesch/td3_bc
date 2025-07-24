# 🧠 Offline to Online TD3-BC

A repository for offline-to-online reinforcement learning using the TD3-BC algorithm. [\[1\]](#ref_td3_bc)

---

## 🚀 First-Time Setup

Follow these steps to set up your development environment.

### ✅ Requirements

* [`uv`](https://github.com/astral-sh/uv) for managing the virtual environment
* `gdown` for downloading pretrained weights (optional, only needed for FTD support):

  ```bash
  python -m pip install gdown
  ```

---

### 🧩 Optional: Install MobileSAMv2 for FTD

If you plan to use TD3-BC with **FTD (Focus-Then-Decide)** [\[2\]](#ref_ftd):

1. Go to the parent directory of `td3_bc`:

   ```bash
   cd ..
   ```

2. Clone the MobileSAM repository:

   ```bash
   git clone https://github.com/ChaoningZhang/MobileSAM.git
   ```

3. Navigate into the MobileSAMv2 directory:

   ```bash
   cd MobileSAM/MobileSAMv2
   ```

4. Download the weights and unzip:

   ```bash
   gdown https://drive.google.com/uc?id=1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE
   unzip weight.zip
   rm -f weight.zip
   ```

5. Apply a patch for weight loading:

   ```bash
   sed -i "s/return torch.load(file, map_location='cpu'), file/return torch.load(file, map_location='cpu', weights_only=False), file/" ultralytics/nn/tasks.py
   ```

---

### 🧠 Install td3-bc

1. Clone the repository:

   ```bash
   git clone https://github.com/maxdoesch/td3_bc.git
   cd td3_bc
   ```

2. Create a virtual environment using `uv`:

   ```bash
   uv venv
   ```

3. Install the dependencies:

   * For core TD3-BC only:

     ```bash
     uv sync
     ```
   * For all optional extras (`gen-expert`, `dm-control`, `ftd`):

     ```bash
     uv sync --all-extras
     ```
   * For specific optional extras:

     ```bash
     uv sync --extra <dependency>
     ```
---

## 🛠️ Usage Guide

### ✅ Activate the Virtual Environment

Before running any scripts, activate the virtual environment:

```bash
source .venv/bin/activate
```

### 🧩 (Optional) Enable FTD Support

If you're using **FTD (Focus-Then-Decide)**, make sure to add **MobileSAMv2** to your `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:../MobileSAM/MobileSAMv2"
```

### 🏋️‍♀️ Start Training

Launch training using one of the available configuration files:

```bash
python train.py --config_path config/config_pretrain.yaml
```

### 🧪 Evaluate a Trained Model

To evaluate a model from a specific checkpoint, run:

```bash
python evaluate.py --checkpoint_path ./<checkpoint_dir>/pretrain --checkpoint <checkpoint_step>
```

Replace `<checkpoint_dir>` with the path to your saved model directory and `<checkpoint_step>` with the specific checkpoint number (e.g., `100000`).

---

### 📁 Available Configurations

* `config/config_pretrain.yaml` — Pretraining phase (offline)
* `config/config_refine.yaml` — Refinement phase (fine-tuning)
* `config/config_online.yaml` — Online learning phase

## 📌 Notes

* Ensure [`uv`](https://github.com/astral-sh/uv) is installed before proceeding.


## 📊 Results

We evaluate **TD3-BC** across both offline and online learning phases using the [Minari](https://github.com/Farama-Foundation/Minari) datasets on **Hopper** and **HalfCheetah** environments. The hyperparameters were selected as stated in [\[1\]](#ref_td3_bc). Each model is evaluated over multiple seeds and reported as *mean ± std* normalized scores. The improvements from each training stage are highlighted in green (positive) or red (negative).

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>pretrain</th>
      <th>refine</th>
      <th>online</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>halfcheetah-simple</td>
      <td>0.65 ± 0.01</td>
      <td>0.64 ± 0.01 (<span style='color:red'>–0.00</span>)</td>
      <td><b>0.75 ± 0.01</b> (<span style='color:green'>+0.10</span>)</td>
    </tr>
    <tr>
      <td>halfcheetah-medium</td>
      <td>0.99 ± 0.16</td>
      <td>0.97 ± 0.15 (<span style='color:red'>–0.03</span>)</td>
      <td><b>1.03 ± 0.02</b> (<span style='color:green'>+0.06</span>)</td>
    </tr>
    <tr>
      <td>halfcheetah-expert</td>
      <td>0.56 ± 0.19</td>
      <td>0.53 ± 0.20 (<span style='color:red'>–0.03</span>)</td>
      <td><b>0.94 ± 0.03</b> (<span style='color:green'>+0.41</span>)</td>
    </tr>
    <tr>
      <td>hopper-simple</td>
      <td>0.98 ± 0.00</td>
      <td>0.98 ± 0.00 (<span style='color:green'>+0.00</span>)</td>
      <td><b>1.00 ± 0.01</b> (<span style='color:green'>+0.02</span>)</td>
    </tr>
    <tr>
      <td>hopper-medium</td>
      <td>1.10 ± 0.01</td>
      <td><b>1.11 ± 0.01</b> (<span style='color:green'>+0.01</span>)</td>
      <td>0.75 ± 0.28 (<span style='color:red'>–0.36</span>)</td>
    </tr>
    <tr>
      <td>hopper-expert</td>
      <td><b>1.07 ± 0.23</b></td>
      <td>0.93 ± 0.25 (<span style='color:red'>–0.14</span>)</td>
      <td>0.70 ± 0.35 (<span style='color:red'>–0.23</span>)</td>
    </tr>
  </tbody>
</table>

## 📚 Reference

This implementation is inspired by the methods proposed in:

<a name="ref_td3_bc">\[1]</a> A. Beeson and G. Montana, “Improving TD3-BC: Relaxed Policy Constraint for Offline Learning and Stable Online Fine-Tuning,” *arXiv preprint arXiv:2211.11802*, 2022. [https://doi.org/10.48550/arXiv.2211.11802](https://doi.org/10.48550/arXiv.2211.11802) 

<a name="ref_ftd">\[2]</a> C. Chen et al., “Focus-Then-Decide: Segmentation-Assisted Reinforcement Learning,” *AAAI 2024*, [https://doi.org/10.1609/aaai.v38i10.29002](https://doi.org/10.1609/aaai.v38i10.29002)
