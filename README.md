# 🎓 Efficient Coreset-Based Neural Architecture Search for Continual Learning
**Reproduction of CLEAS-C (Gao et al., 2022) + Proposed EC-NAS Method**
This repository contains a complete reproduction of the **CLEAS-C** method (Gao et al., 2022) along with an improved and computationally efficient variant, **EC-NAS**, which performs Neural Architecture Search (NAS) using only a small, representative subset of training samples selected via reinforcement learning.
The project was developed as part of the **Reinforcement Learning Research Course** at **Vidyashilp University, Bangalore.**

### The codebase supports two modes:
**CLEAS-C (Paper Baseline)** — Standard architecture search on full training data
**Proposed EC-NAS** — Architecture + Data-Ratio Joint Optimization
Coreset selected using Cosine Prototype Selection (final method)
Also includes K-means clustering coreset implementation for comparison.

The goal of the project is two-fold:
**1. Reduce NAD compute time during continual learning** using reinforcement learning
**2. Perform NAS using only a small subset of representative samples** selected by the RL agent

#### 📌Table of Contents
1. Overview
2. Problem Statement
3. Dataset
4. Methodology
5. CLEAS-C Baseline
6. Proposed EC-NAS Method
7. Training Hyperparameters
8. Evaluation Metrics
9. Implementation Details
10. Repository Structure
11. Running the Code
12. Compute Requirements
13. Experimental Results
14. Key Findings
15. Reproducibility Notes
16. Troubleshooting
17. Citation
18. Authors
    
## 🔍 Overview
Continual learning requires neural networks to learn a sequence of tasks **without forgetting earlier tasks.** The CLEAS-C method addresses this by using **neuron-level architecture search,** guided by a reinforcement learning (RL) controller.
However, CLEAS-C is computationally expensive because:
- Each NAS episode trains a full candidate model
- Every episode uses the entire task dataset
- The original paper uses **200 episodes per task**
  
Our work reproduces CLEAS-C and proposes **EC-NAS,** a more efficient method using:
- Representative coreset sampling
- Cosine prototype selection
- Joint architecture + data-ratio prediction via RL

## 💡 Problem Statement
We address three major challenges in continual learning:
**1. Catastrophic Forgetting** - Retaining performance on old tasks while learning new ones.
**2. Network Capacity Management**- Deciding when to reuse existing neurons vs. expand architecture.
**3. Computational Efficiency** - Reducing the heavy NAS cost present in CLEAS-C.
Our proposed EC-NAS method improves the efficiency of architecture search while maintaining competitive accuracy and significantly reducing forgetting.
  
## 🗂 Dataset
We use **CIFAR-100:**
- 60,000 images
- 100 classes
- 50,000 training / 10,000 testing
- 32×32 RGB images
**Task Split:**
100 classes → 10 tasks → each with 10 classes (Task 0 = 0–9, Task 1 = 10–19, …, Task 9 = 90–99)
Each task is split into:
- 90% training
- 10% validation
- Full test set for evaluation

## 🧠 Methodology
The system consists of four core modules:
**1. Task Loader** – Creates 10 sequential tasks
**2. Task Network** – CNN with progressive expansion
**3. Controller (RL)** – LSTM that selects neuron-level actions
**4. Coreset Selector (Proposed)** – Extracts representative data subset

**Task Network (CLEASC_TaskNetwork)**
The CLEAS paper introduces a neuron-level architecture search method using reinforcement learning (RL) to select and Handles:
- neuron-level reuse/drop/extend
- kernel embedding during expansion
- frozen old weights for forward transfer
- multiple task-specific output heads
However, CLEAS is computationally expensive because:
    ✔ It trains a candidate model per episode
    ✔ Each episode uses the full dataset
    ✔ The paper uses 200 episodes per task (H = 200)
This results in ~20 hours of computation for CIFAR-100 even on GPU.

**Controller (RL)**
Two versions:
- `ControllerLSTM` — original CLEAS-C controller
- `ControllerLSTM_WithRatio` — EC-NAS joint architecture + data ratio controller

## 🧩 CLEAS-C Baseline (Paper Method
CLEAS-C performs neuron-level architecture search using an LSTM Controller.
For each neuron, the controller selects one of four actions:
**1. Drop (0)**
**2. Use (1)**
**3. Drop + Extend (2)**
**4. Use + Extend (3)**

#### 🔄 Architecture Search
For each task t:
- Controller samples **H = 50** candidate architectures
- Each architecture trains newly activated parameters for **8 epochs**
- Performance on validation set gives reward
- Controller updated using **REINFORCE** policy gradient

### 🎯 Reward Function
R = mean_accuracy_all_tasks – α * new_neurons
where α = 0.002.

### 🏗 Weight Transfer
- Reused neurons load weights from previous task and are frozen
- Dropped neurons removed from the architecture
- Newly activated neurons trained from scratch
- Kernel expansion handled by embedding old kernel into larger kernel

## 🚀 Proposed Method (EC-NAS)
The proposed approach improves CLEAS-C through coreset-based training and joint optimization.

### 1️⃣ Representative Data Coresets
We reduce training load by using only a small subset of each task’s data.
**1.1 K-Means Coreset (Ablation)**
- Per-class MiniBatchKMeans clustering
- Samples chosen from each cluster (centroid + diverse farthest samples)
- Helps evaluate whether geometric diversity helps reduce forgetting
- Drawback: slow feature clustering and higher compute

**1.2 Cosine Prototype Coreset (Final method – fastest and better performing)**
- Computes per-class cosine prototypes
- Selects samples closest to class prototypes
- Requires only matrix multiplication → extremely fast
- Produces the best accuracy-forgetting tradeoff
- Reduces search time per task by ~35–40%

### 2️⃣ Joint Architecture + Data Ratio Controller
We extend the controller:
- Predict neuron actions
- Predict data ratio ∈ [ratio_min, ratio_max]
Ratio predicted as:
ratio = ratio_min + sigmoid(h_T) * (ratio_max – ratio_min)
Used to select coreset size.

### 3️⃣ Warmup Phase
First 5 episodes of each task use full training data
Stabilizes early RL gradients

### 4️⃣ Beta Prior Regularization
We apply a Beta(2,5) prior encouraging:
- Lower ratios (higher efficiency)
- Avoid collapse to ratio = 1.0

## ⚙️ Training Hyperparameters
| Parameter	| Value|
|------------|---------|
| Episodes per task (H) | 50 |
| Exploration (p)	| 0.30 |
| Task 1 epochs | 40 |
| Candidate training epochs | 8 |
| Batch size | 64 |
| Task network LR | 1e−3 |
| Controller LR | 7e−4 (RMSprop) |
| Coreset samples per class  | 5 |
| Ratio range | 0.30–0.60 |

## 📏 Evaluation Metrics
**Primary Metrics**
-Per-task accuracy (after learning)
- Final per-task accuracy
- Mean accuracy
- Average forgetting

**Secondary Metrics**
- New neurons added per task
- Data ratios learned
- Search time per task
- Final model size

## 🛠 Implementation Details
**✔ Checkpointing**
Each method stores:
- actions_t.npy
- states_t.pth
- filter_sizes_t.npy
- ratios_t.npy (proposed)
- times_t.npy
- progress.json
Training resumes automatically.

**✔ Feature Caching**
- Features extracted once per task
- Saves 70–80% coreset selection time
- Critical for fast K-Means and prototype computation

**Cosine prototype selection was shown to achieve lower computation + lower forgetting**

## 📁 Repository Structure
    RL_Project/
    │
    ├── CLEAS-C+KMeans_Proposed.ipynb          # abalation     
    ├── CLEAS-C+Cosine_EC-NAS-Proposed.ipynb   # final results    
    │
    └── README.md                # This file

Both notebooks contain both:
- CLEAS-C baseline
- Proposed EC-NAS variant

## ⚙️ Installation & Requirements
**Step 1** — Clone 
    git clone https://github.com/your_repo_here
    cd your_repo_here 

**Step 2** — Create environment
    conda create -n cleas python=3.10 -y
    conda activate cleas 

**Step 3** — Install dependencies
    pip install torch torchvision torchaudio
    pip install numpy scipy scikit-learn
    pip install matplotlib seaborn pillow 

##### Preparing CIFAR-100
The dataset will automatically download into ./data at first run:
data/
&nbsp;  CIFAR100/
No further action required.

## ▶️ How To Run the Code
This project contains two Jupyter Notebooks, each implementing one experimental pipeline:
1. CLEAS-C + KMeans (Ablation Study)
   CLEAS-C+KMeans_Proposed.ipynb

2. CLEAS-C + Cosine Prototype EC-NAS (Final Proposed Method)
   CLEAS-C+Cosine_EC-NAS-Proposed.ipynb

Both notebooks include the full pipeline:
- Load CIFAR-100
- Train Task 1 (full training)
- Perform NAS for Tasks 2–10
- Evaluate:
    Immediate per-task accuracy
    Final per-task accuracy
    Forgetting per task
    Mean forgetting
    Search time
- Save all results and checkpoints
- Generate all graphs

#### 🎬 1. Running the Experiments
Everything runs directly from the notebooks.

To run CLEAS-C + KMeans version
Open:
     CLEAS-C+KMeans_Proposed.ipynb
Run all cells (Run All in Colab or jupyter notebook).

This will execute:
- CLEAS-C baseline
- Proposed method with KMeans coreset selection
and produce:
- accuracy tables
- forgetting curves
- search time plots
- saved checkpoints

To run the final EC-NAS with Cosine Prototypes
Open:
     CLEAS-C+Cosine_EC-NAS-Proposed.ipynb
Run all cells.
This notebook implements:
- CLEAS-C baseline
- Final proposed EC-NAS
- Cosine prototype selection (recommended)
- Joint architecture + data-ratio controller
- Faster coreset selection and lower forgetting
  
#### 🧮 2. Compute Requirements
| Resource | Requirement                          |
| -------- | ------------------------------------ |
| GPU      | Recommended (RTX 2060+, T4, A100)    |
| RAM      | 16GB+                                |
| Disk     | 2GB                                  |
| Runtime  | CLEAS-C ~10–15 hrs, EC-NAS ~6–10 hrs |


#### 💾 3. Checkpointing (Automatic)
Each experiment automatically creates:

    /checkpoints_cleasc/
        actions_t.npy
        states_t.pth
        filter_sizes_t.npy
        times_t.npy
        progress.json

    /checkpoints_proposed/
        actions_t.npy
        states_t.pth
        filter_sizes_t.npy
        ratios_t.npy
        times_t.npy
        progress.json
These are used to automatically resume where the notebook left off. It allow notebook to resume mid-experiment even after runtime disconnection.

Resume from checkpoint
Just open the notebook and run all cells again.
It will detect:
progress.json → last_completed_task
and continue from that point.

#### 📊 4. Outputs Generated
Each notebook produces:
- cleasc_results.pt
- proposed_results.pt
- Immediate per-task accuracy
- Final accuracy
- Forgetting per task
- Mean forgetting
- Search-time and total-time plots
- New neurons per task
- Cumulative accuracy curves

These plots allow direct comparison between:
    CLEAS-C
    KMeans EC-NAS
    Cosine EC-NAS

#### 5. 📈 Experimental Results
**⭐ K-Means Coreset Results (Ablation)**
- Forgetting reduced by 73%
- But accuracy drops due to noise in clustering
- Search time ↑ due to clustering cost

**⭐ Cosine Prototype EC-NAS (Final Method)**
- After-learning accuracy +73%
- Final accuracy +35%
- Forgetting reduced
- Search time per task −36–40%
- Most consistent performance across tasks

**Overall:**
**Cosine prototype EC-NAS outperforms both CLEAS-C and K-Means variants.**

#### 📈 6. Results Summary
##### 6.1 Accuracy
Cosine EC-NAS significantly improves:
- After-learn accuracy (+73%)
![WhatsApp Image 2025-11-30 at 15 14 40_856223e9](https://github.com/user-attachments/assets/17995620-e88e-4638-9f9b-b34a770929d0)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Final accuracy (+35%)
![WhatsApp Image 2025-11-30 at 15 14 40_37065442](https://github.com/user-attachments/assets/2635882d-5d2b-4f3e-ac32-df3917d1651b)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

- New neuron per task
![WhatsApp Image 2025-11-30 at 15 14 41_d2e84fa1](https://github.com/user-attachments/assets/e3101d07-51a2-49c8-9e04-966793596ce8)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

##### 6.2 Forgetting
Cosine EC-NAS drastically increase catastrophic forgetting.
![WhatsApp Image 2025-11-30 at 15 19 27_da18d1cf](https://github.com/user-attachments/assets/848f0bf5-34be-4565-b547-f0fd467ac6f8)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
KMeans reduces some forgetting but suffers in accuracy.
![WhatsApp Image 2025-11-30 at 15 20 57_8f71ee49](https://github.com/user-attachments/assets/f813c5bb-75d6-48ba-93dd-6bf501446798)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

##### .3 Compute Efficiency
Cosine EC-NAS reduces:
- per-task search time by ~37%
- overall memory + compute footprint
  ![WhatsApp Image 2025-11-30 at 15 14 40_31b0fbaf](https://github.com/user-attachments/assets/c3e47ddc-1a76-47a8-8235-45e3b64f35b6)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

##### 6.4 Overall Conclusion
Cosine-based EC-NAS is superior in both efficiency and stability.
KMeans is included only for ablation and interpretability.

#### 🧠 7. Key Findings
EC-NAS substantially reduces computation cost versus CLEAS-C 
Training per task becomes 35–40% faster.
Cosine prototype coreset outperforms K-Means in both speed and stability
Data ratio prediction improves sample efficiency
Architecture grows more compactly
Forgetting is reduced through coreset-based training
Per-episode training cost drops significantly

#### 🔁 8. Reproducibility Notes
To match CLEAS-C paper more closely:
- Increase episodes H = 200 instead of 50
- Use ratio_min = 0.65, ratio_max = 0.9
- Train candidate with SGD
- Warm-up for first 10 episodes

#### 🛠 9. Troubleshooting
**CUDA Out of Memory**
- Reduce batch size:"batch_size": 32
**KMeans too slow**
- Use cosine method (recommended).
Mismatch between CLEAS-C and Proposed

Clear checkpoints:
rm -rf checkpoints_cleasc/*
rm -rf checkpoints_proposed/*

#### 📚 10. Citation
If you use this implementation, please cite:
Gao, Q., Luo, Z., \& Klabjan, D. (2022).
Efficient Architecture Search for Continual Learning.

#### 🙌 11. Acknowledgements
This project was developed for academic research on neural architecture search and efficient coreset-based continual learning.
Feel free to reach out or open issues for support or clarification.

#### 🧑‍💻 Author Information
- **Team:** Anchal Gupta, Nikita Agre, Rajiv Jarhad
- **Course:** Reinforcement Learning Course
- **Institution:** Vidyashilp University, Bangalore 
- **Date:** Dec 2025
