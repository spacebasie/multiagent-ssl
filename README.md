# Multi-Agent Decentralized Self-Supervised Learning (VICReg)

This repository contains the implementation of a decentralized, multi-agent representation learning framework using **VICReg** (Variance-Invariance-Covariance Regularization). The project investigates how agents can learn high-quality representations and classifiers under various non-IID conditions (Label Skew and Domain Shift) and network topologies without sharing raw data.

This repo contains the entire codebase of my MSc Thesis for Imperial: "Representation Learning for Collective Intelligence".

To read my thesis or poster, check out the respective page on my [website](https://spacebasie.github.io/research/)

## 📋 Table of Contents
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Running Simulations](#-running-simulations)
    - [1. Centralized Baseline](#1-centralized-baseline)
    - [2. Federated Learning](#2-federated-learning)
    - [3. Decentralized Learning (Personalized)](#3-decentralized-learning-personalized)
    - [4. Alignment & Collaborative Classification (Novel Method)](#4-alignment--collaborative-classification-novel-method)
- [Logging & Visualization](#-logging--visualization)

---

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd multiagent-ssl
   
2. **Install Dependencies:** Ensure you have Python 3.10+ installed. Install the required packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt

## 📂 Project Structure

### Core Execution

* `main.py`: The primary entry point. Parses command-line arguments and orchestrates the training loops for Centralized, Federated, and Decentralized modes.
* `config.py`: Contains global hyperparameters (learning rates, batch sizes, VICReg coefficients, network constants).
* `pbs.sh`: A Bash script for submitting jobs to an HPC cluster (PBS scheduler).

### Models & Components

* `model.py`: Defines the `VICReg` architecture (Backbone + Projection Head) and the `VICRegLoss` function.
* `network.py`: Handles graph topology generation (Ring, Random, Fully Connected, Disconnected) and Gossip Averaging logic.
* `data.py`: Handles loading and transformations for **CIFAR-10/100**.
* `custom_datasets.py`: Handles loading for **Office-Home**, including domain-specific splitting and hierarchical setups.
* `data_splitter.py`: Logic for creating non-IID partitions (Dirichlet Label Skew) and applying Domain Shifts.

### Training Loops
* `training.py`: Functions for Centralized training and Standard Federated local updates/aggregation.
* `decentralized_training.py`: Logic for **Personalized Decentralized Learning**. Evaluates agents on both their local skewed test sets and global test sets.
* `combo_training.py`: Implements the **Novel Method** (Representation Alignment + Collaborative Classifier Consensus).

### Evaluation & Plotting
* `evaluate.py`: Protocols for Linear Probing and k-Nearest Neighbors (k-NN) evaluation. Also contains t-SNE and PCA visualization tools.
* `plot_res.py`: A script to parse CSV logs from `simulations/` and generate comparative plots (Learning curves, Angle evolution, Loss terms).

---

## ⚙️ Configuration

You can adjust hyperparameters in `config.py`, but most key parameters can be overridden via command-line arguments in `main.py`.

**Key Arguments:**

* `--mode`: `centralized`, `federated`, or `decentralized`.
* `--dataset`: `cifar10` or `office_home`.
* `--num_agents`: Number of agents in the network.
* `--topology`: `fully_connected`, `ring`, `random`, or `disconnected`.
* `--heterogeneity_type`: Defines data split (e.g., `label_skew`, `domain_shift`, `combo_domain`).
* `--alpha`: Dirichlet parameter for label skew (Lower = More Heterogeneous).
* `--alignment_strength`: Weight for the feature alignment loss (for the Novel Method).

------------------------------------------------------------------------

## 🚀 Running Simulations

### 1. Centralized Training

``` bash
python main.py --mode centralized --dataset cifar10 --epochs 200 --eval_every 5
```

### 2. Federated Learning (FedAvg)

``` bash
python main.py --mode federated --dataset cifar10 --num_agents 5 --alpha 0.5 --comm_rounds 200
```

### 3. Decentralized Learning (Personalized)

#### CIFAR-10 (Label Skew)

``` bash
python main.py   --mode decentralized   --dataset cifar10   --heterogeneity_type label_skew_personalized   --topology random   --num_agents 10   --alpha 0.5   --comm_rounds 200
```

#### Office-Home (Domain Split)

``` bash
python main.py   --mode decentralized   --dataset office_home   --heterogeneity_type office_domain_split   --topology fully_connected   --num_agents 4   --comm_rounds 200
```

### 4. Novel Method: Alignment + Collaborative Classification

#### Office-Home

``` bash
python main.py   --mode decentralized   --dataset office_home   --heterogeneity_type combo_domain   --topology random   --num_agents 8   --alignment_strength 25   --comm_rounds 200   --eval_every 5
```

#### CIFAR-10

``` bash
python main.py   --mode decentralized   --dataset cifar10   --heterogeneity_type combo_label_skew   --topology ring   --num_agents 5   --alpha 0.1   --alignment_strength 10
```

------------------------------------------------------------------------

## 📊 Logging & Visualization

### Weights & Biases
The project automatically logs metrics to WandB

* **Metrics Logged:** Training Loss (Invariance, Variance, Covariance), k-NN Accuracy, Linear Eval Accuracy, Representation Angles (misalignment).
* **Visuals:** t-SNE plots and PCA plots are generated periodically and at the end of training.

### Local CSV Logging
Training statistics are saved to the `simulations/` folder as CSV files. You can generate publication-ready plots using the provided script:

``` bash
# Edit plot_res.py to select which plot to generate (e.g., plot = 'learning_curve')
python plot_res.py
```

Generated figures will be saved to the `figures/` directory.

