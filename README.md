# 🎓 Efficient Coreset-Based Neural Architecture Search for Continual Learning
This repository contains a complete reproduction of the CLEAS-C method (Gao et al., 2022) along with an improved and computationally efficient variant, EC-NAS, which performs Neural Architecture Search (NAS) using only a small, representative subset of training samples selected via reinforcement learning.
The project is built as part of a university research course in Reinforcement Learning at Vidyashilp University, Bangalore.

### The codebase supports two modes:
**CLEAS-C (Paper Baseline)** — Standard architecture search on full training data
**Proposed EC-NAS** — Architecture + Data-Ratio Joint Optimization
Coreset selected using Cosine Prototype Selection (final method)
Also includes K-means clustering coreset implementation for comparison.

## 🔍 Project Overview
Continual learning systems must adapt to a sequence of tasks while avoiding catastrophic forgetting.
The project is implemented in modular components aligned with the CLEAS framework:
**Dataset & Task Loader** 
- CIFAR-100 loaded once
- Split into **10 disjoint tasks**, each containing 10 classes
- Each task has train/val/test loaders

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

###  🚀 Our Proposed Method (EC-NAS)
We improve efficiency by introducing:
##### 1️⃣. Representative Coreset Selection
Rather than training each episode on full task data, we choose small representative subsets per task.
We implement two coreset extraction mechanisms used during the NAS search.

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

##### 2️⃣. Data Ratio Prediction by RL Controller
The controller joint-optimizes:
    architecture actions
    global data ratio
    ratio∈[ratio_min,ratio_max]

Ratio is predicted as:
ratio = ratio_min + sigmoid(ratio_head(h_T)) * ratio_range

##### 3️⃣. Joint Architecture + Data Optimization
REINFORCE now optimizes for:
- higher average accuracy on all tasks
- smaller architectural expansion
- lower forgetting
- efficient coreset selection

##### 4️⃣. Improved State Encoding + Layer One-Hot
Each neuron state = one\_hot(action) + layer\_encoding allowing better RL learning signals.

##### 5️⃣. Warm-up \& ratio constraints
Task 1 always uses full data (ratio = 1.0).
For later tasks, ratio is constrained:
ratio = ratio\_min + sigmoid(ratio\_head(h\_T)) \* ratio\_range

**Cosine prototype selection was shown to achieve lower computation + lower forgetting**

### 📁 Repository Structure
    RL_Project/
    │
    ├── CLEAS-C+KMeans_Proposed.ipynb          # abalation     
    ├── CLEAS-C+Cosine_EC-NAS-Proposed.ipynb   # final results    
    │
    ├── checkpoints_cleasc/      # Auto-generated CLEAS-C task checkpoints
    ├── checkpoints_proposed/    # Auto-generated proposed method checkpoints
    │
    └── README.md                # This file

Both notebooks contain both:
- CLEAS-C baseline
- Proposed EC-NAS variant

### ⚙️ Installation & Requirements
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

### ▶️ How To Run the Code
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

#### 💾 2. Checkpointing (Automatic)
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

#### 📊 3. Outputs Generated
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

#### 📈 4. Results Summary
##### 4.1 Accuracy
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

##### 4.2 Forgetting
Cosine EC-NAS drastically increase catastrophic forgetting.
![WhatsApp Image 2025-11-30 at 15 19 27_da18d1cf](https://github.com/user-attachments/assets/848f0bf5-34be-4565-b547-f0fd467ac6f8)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
KMeans reduces some forgetting but suffers in accuracy.
![WhatsApp Image 2025-11-30 at 15 20 57_8f71ee49](https://github.com/user-attachments/assets/f813c5bb-75d6-48ba-93dd-6bf501446798)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

##### 4.3 Compute Efficiency
Cosine EC-NAS reduces:
- per-task search time by ~37%
- overall memory + compute footprint
  ![WhatsApp Image 2025-11-30 at 15 14 40_31b0fbaf](https://github.com/user-attachments/assets/c3e47ddc-1a76-47a8-8235-45e3b64f35b6)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

##### 4.4 Overall Conclusion
Cosine-based EC-NAS is superior in both efficiency and stability.
KMeans is included only for ablation and interpretability.

#### 🧠 5. Key Findings
EC-NAS substantially reduces computation cost versus CLEAS-C
Cosine prototype coreset outperforms K-Means in both speed and stability
Data ratio prediction improves sample efficiency
Architecture grows more compactly
Forgetting is reduced through coreset-based training
Per-episode training cost drops significantly

#### 🔁 6. Reproducibility Notes
To match CLEAS-C paper more closely:
- Increase episodes H = 200 instead of 50
- Use ratio_min = 0.65, ratio_max = 0.9

Optional:
Use SGD for candidate model training
Warm-up first 10 episodes with ratio = 1.0

#### 🛠 7. Troubleshooting
CUDA Out of Memory
Reduce batch size:"batch_size": 32
KMeans too slow
Use cosine method (recommended).
Mismatch between CLEAS-C and Proposed

Clear checkpoints:
rm -rf checkpoints_cleasc/*
rm -rf checkpoints_proposed/*

#### 📚 8. Citation
If you use this implementation, please cite:
Gao, Q., Luo, Z., \& Klabjan, D. (2022).
Efficient Architecture Search for Continual Learning.

#### 🙌 9. Acknowledgements
This project was developed for academic research on neural architecture search and efficient coreset-based continual learning.
Feel free to reach out or open issues for support or clarification.

#### 🧑‍💻 Author Information
- **Team:** Anchal Gupta, Nikita Agre, Rajiv Jarhad
- **Course:** Reinforcement Learning Course
- **Institution:** Vidyashilp University, Bangalore 
- **Date:** Dec 2025
