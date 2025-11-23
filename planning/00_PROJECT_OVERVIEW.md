# AI for Trustworthy Decision Making - Project Plan
## COVIDx CXR-4 Dataset | Team of 3

---

## Project Overview

| Attribute | Details |
|-----------|---------|
| **Dataset** | COVIDx CXR-4 (Chest X-ray images for COVID-19 detection) |
| **Framework** | Flower (Federated Learning) |
| **Trust Dimensions** | Privacy (Differential Privacy) + Interpretability (Grad-CAM/SHAP) |
| **Total Points** | 6 (Stage 1: 2pts, Stage 2: 4pts) |

---

## Team Role Assignment

| Role | Member | Primary Focus |
|------|--------|---------------|
| **Member 1** | Data & Experiment Design Lead | Dataset, preprocessing, federated setup |
| **Member 2** | Modeling & Privacy Lead | Model architecture, Differential Privacy |
| **Member 3** | Evaluation & Interpretability Lead | Grad-CAM, SHAP, robustness testing |

---

# STAGE 1: Baseline Development & Minimal Federated Setup (2 points)

## Member 1: Data & Experiment Design Lead

### Actionables
1. **Download and explore COVIDx CXR-4 dataset**
   - Download from official source
   - Understand data structure (images, labels, metadata)
   - Document class distribution (COVID-positive, COVID-negative, normal, pneumonia)

2. **Design data preprocessing pipeline**
   - Image resizing (e.g., 224x224 for standard CNNs)
   - Normalization strategy
   - Data augmentation (rotation, flipping, contrast adjustment)
   - Handle class imbalance (oversampling/undersampling/class weights)

3. **Create federated data splits**
   - Split dataset into 3 client partitions
   - Implement **non-IID splits** (realistic scenario)
   - Option A: Label skew (each client has different class distributions)
   - Option B: Quantity skew (different number of samples per client)
   - Document partitioning strategy

4. **Implement data loading utilities**
   - Create PyTorch Dataset class for COVIDx CXR-4
   - Create data loader scripts for each client
   - Script to generate `client_1_data.npz`, `client_2_data.npz`, `client_3_data.npz`

5. **Set up project infrastructure**
   - Initialize Git repository structure
   - Set up virtual environment with dependencies
   - Create `requirements.txt`
   - Write data download/preprocessing scripts

### Deliverables
| Deliverable | Format | Description |
|-------------|--------|-------------|
| `data/` folder structure | Directory | Organized raw and processed data |
| `src/data/preprocessing.py` | Python | Image preprocessing pipeline |
| `src/data/dataset.py` | Python | PyTorch Dataset class |
| `src/data/split_federated.py` | Python | Script to create non-IID client splits |
| `client_X_data.npz` files | NPZ | Pre-split data for each client |
| Data exploration notebook | Jupyter | EDA with visualizations |
| `requirements.txt` | Text | All project dependencies |
| Stage 1 Report Section | Doc | Dataset description, preprocessing pipeline, partitioning strategy |

---

## Member 2: Modeling & Privacy Lead

### Actionables
1. **Research and select base model architecture**
   - Survey CNN architectures for medical imaging
   - Options: ResNet18, EfficientNet-B0, DenseNet121, custom CNN
   - Consider model size for federated setting (communication cost)

2. **Implement centralized baseline model**
   - Create model class in PyTorch
   - Implement training loop (optimizer, loss function, scheduler)
   - Train on full aggregated dataset
   - Save model weights and training logs

3. **Implement Flower server**
   - Set up `server.py` with FedAvg strategy
   - Configure federation rounds and client participation
   - Implement metric aggregation functions
   - Test server startup

4. **Implement Flower client**
   - Create `client.py` with NumPyClient
   - Implement `get_parameters()`, `set_parameters()`, `fit()`, `evaluate()`
   - Integrate with data loaders from Member 1
   - Handle local training epochs configuration

5. **Run federated experiments**
   - Test with 3 simulated clients on single machine
   - Compare FedAvg vs centralized baseline
   - Document training curves and convergence

### Deliverables
| Deliverable | Format | Description |
|-------------|--------|-------------|
| `src/models/cnn_model.py` | Python | Base CNN architecture |
| `src/models/train_centralized.py` | Python | Centralized training script |
| `src/federated/server.py` | Python | Flower server implementation |
| `src/federated/client.py` | Python | Flower client implementation |
| `configs/` | YAML/JSON | Hyperparameter configurations |
| Centralized model weights | `.pth` | Trained baseline model |
| Federated model weights | `.pth` | Model after FL training |
| Training logs | CSV/JSON | Loss, accuracy per round |
| Stage 1 Report Section | Doc | Model architecture, FL setup description |

---

## Member 3: Evaluation & Robustness/Interpretability Lead

### Actionables
1. **Define evaluation metrics**
   - Classification metrics: Accuracy, Precision, Recall, F1-score
   - Medical-specific: Sensitivity, Specificity, AUC-ROC
   - Per-class metrics (COVID vs others)

2. **Implement evaluation framework**
   - Create evaluation script with all metrics
   - Confusion matrix visualization
   - ROC curves and AUC calculation
   - Statistical significance testing

3. **Evaluate centralized baseline**
   - Run evaluation on test set
   - Generate comprehensive metrics report
   - Analyze per-class performance

4. **Evaluate federated baseline**
   - Evaluate global model after FL
   - Evaluate each client's local model
   - Compare centralized vs federated performance

5. **Perform initial robustness assessment**
   - Test model on perturbed inputs (Gaussian noise, blur)
   - Document sensitivity to input variations
   - Identify potential weaknesses for Stage 2

6. **Prepare Stage 1 presentation**
   - Consolidate results from all members
   - Create visualizations and slides
   - Identify next steps for Stage 2

### Deliverables
| Deliverable | Format | Description |
|-------------|--------|-------------|
| `src/evaluation/metrics.py` | Python | Evaluation functions |
| `src/evaluation/evaluate.py` | Python | Main evaluation script |
| `results/stage1/` | Directory | All evaluation results |
| Confusion matrices | PNG | Visual confusion matrices |
| ROC curves | PNG | AUC-ROC visualizations |
| Comparison table | CSV/Markdown | Centralized vs Federated metrics |
| Stage 1 Report Section | Doc | Preliminary results, limitations, trust dimensions for Stage 2 |
| Stage 1 Presentation | Slides | Team presentation |

---

# STAGE 2: Trustworthiness Enhancements (4 points)

## Member 1: Data & Experiment Design Lead (Stage 2)

### Actionables
1. **Create diverse data splits for cross-evaluation**
   - Design 3 distinct dataset variants
   - Variant A: Different label distributions
   - Variant B: Different image quality/sources
   - Variant C: Temporal split (if metadata available)

2. **Implement data poisoning attacks (for robustness testing)**
   - Label flipping attack
   - Backdoor attack (trigger patterns)
   - Document attack implementations

3. **Support privacy experiments**
   - Integrate with DP-enhanced training from Member 2
   - Measure utility loss under different privacy budgets

4. **Cross-evaluation coordination**
   - Share preprocessed datasets with team
   - Run Member 2's and Member 3's models on your data variant
   - Document cross-client performance

5. **Final data documentation**
   - Complete data card for COVIDx CXR-4
   - Document all preprocessing decisions
   - Ethical considerations of dataset usage

### Deliverables (Stage 2)
| Deliverable | Format | Description |
|-------------|--------|-------------|
| Multiple dataset variants | NPZ | 3 distinct data splits |
| `src/attacks/data_poisoning.py` | Python | Attack implementations |
| Cross-evaluation results | CSV | Results on all models |
| Data card | Markdown | Complete dataset documentation |
| Report sections (6 pages) | Doc | Data analysis, cross-evaluation |

---

## Member 2: Modeling & Privacy Lead (Stage 2)

### Actionables
1. **Implement Differential Privacy (DP-SGD)**
   - Integrate Opacus library with PyTorch model
   - Implement gradient clipping
   - Add noise to gradients
   - Configure privacy budget (epsilon, delta)

2. **Run privacy experiments**
   - Train models with different epsilon values (1, 5, 10, ∞)
   - Measure accuracy vs privacy trade-off
   - Document convergence with DP

3. **Implement DP in federated setting**
   - Local DP at client level
   - Test with Flower framework
   - Compare: No DP vs Local DP vs Central DP

4. **Cross-model evaluation**
   - Share trained models with teammates
   - Evaluate your DP models on Member 1's and Member 3's data
   - Compare DP impact across different data distributions

5. **Privacy analysis**
   - Membership inference attack resistance
   - Document privacy guarantees
   - Visualize privacy-utility trade-off curves

### Deliverables (Stage 2)
| Deliverable | Format | Description |
|-------------|--------|-------------|
| `src/privacy/dp_training.py` | Python | DP-SGD implementation |
| `src/federated/client_dp.py` | Python | DP-enhanced Flower client |
| DP model weights | `.pth` | Models at different epsilon |
| Privacy-utility curves | PNG | Trade-off visualizations |
| Cross-evaluation results | CSV | DP model performance across clients |
| Report sections (6 pages) | Doc | Privacy mechanisms, experiments |

---

## Member 3: Evaluation & Interpretability Lead (Stage 2)

### Actionables
1. **Implement Grad-CAM**
   - Grad-CAM for CNN visualization
   - Generate saliency maps for predictions
   - Analyze what model "sees" in X-rays

2. **Implement SHAP explanations**
   - SHAP values for feature importance
   - Compare explanations across models
   - Document explanation consistency

3. **Robustness testing**
   - Test against data poisoning attacks from Member 1
   - Adversarial example testing (FGSM, PGD)
   - Out-of-distribution detection

4. **Trust metrics evaluation**
   - Define and compute trust metrics
   - Calibration analysis (reliability diagrams)
   - Uncertainty quantification

5. **Comprehensive cross-model evaluation**
   - Evaluate all models (baseline, DP, federated variants)
   - Create comparison tables
   - Statistical significance tests

6. **Final presentation preparation**
   - Consolidate all results
   - Create unified visualizations
   - Coordinate presentation sections

### Deliverables (Stage 2)
| Deliverable | Format | Description |
|-------------|--------|-------------|
| `src/interpretability/gradcam.py` | Python | Grad-CAM implementation |
| `src/interpretability/shap_explain.py` | Python | SHAP explanations |
| `src/robustness/adversarial.py` | Python | Adversarial testing |
| Saliency map gallery | PNG | Grad-CAM visualizations |
| SHAP summary plots | PNG | Feature importance |
| Robustness results | CSV | Attack success rates |
| Final comparison tables | Markdown | All models, all metrics |
| Report sections (6 pages) | Doc | Interpretability, robustness |
| Final presentation | Slides | Team presentation |

---

# Gantt Chart

```
WEEK 1-2: Stage 1 Foundation
══════════════════════════════════════════════════════════════════════════════════
              │ W1-Mon │ W1-Tue │ W1-Wed │ W1-Thu │ W1-Fri │ W2-Mon │ W2-Tue │ W2-Wed │ W2-Thu │ W2-Fri │
──────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
MEMBER 1      │████████│████████│████████│████████│████████│████████│████████│        │        │        │
Data Download │████████│████████│        │        │        │        │        │        │        │        │
Preprocessing │        │████████│████████│████████│        │        │        │        │        │        │
Fed Splits    │        │        │        │████████│████████│████████│        │        │        │        │
DataLoaders   │        │        │        │        │        │████████│████████│        │        │        │
──────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
MEMBER 2      │████████│████████│        │        │████████│████████│████████│████████│████████│        │
Model Research│████████│████████│        │        │        │        │        │        │        │        │
Centralized   │        │        │░░WAIT░░│░░WAIT░░│████████│████████│        │        │        │        │
FL Server     │        │        │████████│████████│        │████████│        │        │        │        │
FL Client     │        │        │        │        │████████│████████│████████│        │        │        │
FL Experiments│        │        │        │        │        │        │████████│████████│████████│        │
──────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
MEMBER 3      │████████│████████│████████│        │        │        │████████│████████│████████│████████│
Metrics Def   │████████│████████│        │        │        │        │        │        │        │        │
Eval Framework│        │████████│████████│        │        │        │        │        │        │        │
Eval Central  │        │        │        │░░WAIT░░│░░WAIT░░│░░WAIT░░│████████│████████│        │        │
Eval Federated│        │        │        │        │        │        │        │████████│████████│        │
Report/Present│        │        │        │        │        │        │        │        │████████│████████│
──────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

LEGEND: ████ = Active work  ░░WAIT░░ = Blocked/waiting for dependency

WEEK 3-5: Stage 2 Trust Mechanisms
══════════════════════════════════════════════════════════════════════════════════
              │ W3-Mon │ W3-Wed │ W3-Fri │ W4-Mon │ W4-Wed │ W4-Fri │ W5-Mon │ W5-Wed │ W5-Fri │
──────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
MEMBER 1      │████████│████████│████████│████████│████████│████████│████████│████████│████████│
Data Variants │████████│████████│        │        │        │        │        │        │        │
Poisoning Atk │        │████████│████████│████████│        │        │        │        │        │
Cross-Eval    │        │        │        │████████│████████│████████│████████│        │        │
Documentation │        │        │        │        │        │████████│████████│████████│        │
Report Writing│        │        │        │        │        │        │        │████████│████████│
──────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
MEMBER 2      │████████│████████│████████│████████│████████│████████│████████│████████│████████│
DP-SGD Impl   │████████│████████│████████│        │        │        │        │        │        │
DP Experiments│        │        │████████│████████│████████│        │        │        │        │
DP + FL       │        │        │        │████████│████████│████████│        │        │        │
Cross-Eval    │        │        │        │        │        │████████│████████│████████│        │
Report Writing│        │        │        │        │        │        │        │████████│████████│
──────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
MEMBER 3      │████████│████████│████████│████████│████████│████████│████████│████████│████████│
Grad-CAM      │████████│████████│        │        │        │        │        │        │        │
SHAP          │        │████████│████████│        │        │        │        │        │        │
Robustness    │        │        │████████│████████│████████│        │        │        │        │
All Model Eval│        │        │        │        │████████│████████│████████│        │        │
Final Report  │        │        │        │        │        │        │████████│████████│████████│
Presentation  │        │        │        │        │        │        │        │████████│████████│
──────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

---

# Collaboration Points & Sync Meetings

## Critical Handoff Points

| When | From | To | What | Blocker Risk |
|------|------|-----|------|--------------|
| **W1-Wed** | M1 | M2 | Preprocessed data samples | M2 cannot start centralized training |
| **W1-Thu** | M1 | M2 | Federated splits ready | M2 cannot implement FL client |
| **W2-Mon** | M2 | M3 | Centralized model weights | M3 cannot evaluate |
| **W2-Wed** | M2 | M3 | Federated model weights | M3 cannot compare FL vs central |
| **W3-Mon** | M1 | M2,M3 | Data variants for cross-eval | Stage 2 blocked |
| **W4-Mon** | M1 | M2,M3 | Attack implementations | Cannot test robustness |
| **W4-Wed** | M2 | M1,M3 | DP model weights | Cannot do cross-evaluation |
| **W5-Mon** | ALL | ALL | All models + results | Final integration blocked |

## Recommended Sync Meetings

| Meeting | When | Duration | Agenda |
|---------|------|----------|--------|
| **Kickoff** | W1-Mon | 1hr | Role assignment, setup, questions |
| **Data Ready** | W1-Thu | 30min | M1 demos data, M2 confirms integration |
| **Stage 1 Check** | W2-Wed | 1hr | Review all results, identify gaps |
| **Stage 1 Final** | W2-Fri | 2hr | Finalize report, practice presentation |
| **Stage 2 Kickoff** | W3-Mon | 1hr | Plan Stage 2, confirm trust dimensions |
| **Mid-Stage 2** | W4-Mon | 1hr | Cross-evaluation coordination |
| **Integration** | W5-Mon | 2hr | Combine all results, identify inconsistencies |
| **Final Review** | W5-Thu | 2hr | Complete report, rehearse presentation |

---

# Dependencies & Blockers Diagram

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     STAGE 1 DEPENDENCIES                     │
                    └─────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │   MEMBER 1   │
    │  Data Lead   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐         ┌──────────────┐
    │   Download   │────────►│  Preprocess  │
    │   Dataset    │         │   Pipeline   │
    └──────────────┘         └──────┬───────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
             ┌──────────┐    ┌──────────┐    ┌──────────┐
             │ Client 1 │    │ Client 2 │    │ Client 3 │
             │   Data   │    │   Data   │    │   Data   │
             └────┬─────┘    └────┬─────┘    └────┬─────┘
                  │               │               │
                  └───────────────┼───────────────┘
                                  │
                    ╔═════════════╧═════════════╗
                    ║   BLOCKER: M2 needs data  ║
                    ╚═════════════╤═════════════╝
                                  │
                                  ▼
                         ┌──────────────┐
                         │   MEMBER 2   │
                         │ Modeling Lead│
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │ Centralized│   │   Flower   │   │   Flower   │
       │   Model    │   │   Server   │   │   Client   │
       └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
             │                │                 │
             │                └────────┬────────┘
             │                         │
             │                         ▼
             │                ┌────────────────┐
             │                │   Federated    │
             │                │   Experiments  │
             │                └───────┬────────┘
             │                        │
             └────────────┬───────────┘
                          │
            ╔═════════════╧═════════════╗
            ║  BLOCKER: M3 needs models ║
            ╚═════════════╤═════════════╝
                          │
                          ▼
                 ┌──────────────┐
                 │   MEMBER 3   │
                 │  Eval Lead   │
                 └──────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Eval    │  │  Eval    │  │ Compare  │
    │ Central  │  │ Federated│  │  Results │
    └──────────┘  └──────────┘  └────┬─────┘
                                     │
                                     ▼
                           ┌─────────────────┐
                           │  STAGE 1 DONE   │
                           │ Report + Present│
                           └─────────────────┘


                    ┌─────────────────────────────────────────────────────────────┐
                    │                     STAGE 2 DEPENDENCIES                     │
                    └─────────────────────────────────────────────────────────────┘

         MEMBER 1                    MEMBER 2                    MEMBER 3
    ┌──────────────┐            ┌──────────────┐            ┌──────────────┐
    │ Data Variants│            │   DP-SGD     │            │   Grad-CAM   │
    │ + Attacks    │            │   Training   │            │   + SHAP     │
    └──────┬───────┘            └──────┬───────┘            └──────┬───────┘
           │                           │                           │
           │    ┌──────────────────────┤                           │
           │    │                      │                           │
           ▼    ▼                      ▼                           │
    ╔════════════════╗          ┌──────────────┐                   │
    ║ M2 needs attack║          │  DP Models   │                   │
    ║ implementations║          │  (ε=1,5,10)  │                   │
    ╚════════════════╝          └──────┬───────┘                   │
                                       │                           │
                   ┌───────────────────┴───────────────────┐       │
                   │                                       │       │
                   ▼                                       ▼       ▼
           ┌──────────────┐                        ╔═══════════════════╗
           │   DP + FL    │                        ║ M3 needs all      ║
           │   Combined   │                        ║ models for final  ║
           └──────┬───────┘                        ║ evaluation        ║
                  │                                ╚═════════╤═════════╝
                  │                                          │
                  └──────────────────┬───────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  CROSS-EVALUATION   │
                          │  (All members test  │
                          │   all models)       │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   FINAL INTEGRATION │
                          │   Report (18-20 pg) │
                          │   + Presentation    │
                          └─────────────────────┘
```

---

# Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| COVIDx dataset too large | Medium | High | Use subset, or PneumoniaMNIST as backup |
| Flower version incompatibility | Low | Medium | Pin versions in requirements.txt |
| DP training unstable | Medium | Medium | Start with high epsilon, tune gradually |
| Grad-CAM not working with custom model | Low | Low | Use pretrained ResNet backbone |
| Team member unavailable | Low | High | Cross-train on each other's code |
| GPU not available | Medium | Medium | Use Google Colab Pro or university cluster |
| Non-IID splits too extreme | Medium | Low | Try multiple partitioning strategies |

---

# Repository Structure

```
upb_aitdm_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── training_config.yaml
│   └── federated_config.yaml
├── data/
│   ├── raw/                    # Original COVIDx data
│   ├── processed/              # Preprocessed images
│   └── federated/              # Client splits
│       ├── client_1_data.npz
│       ├── client_2_data.npz
│       └── client_3_data.npz
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Member 1
│   │   ├── dataset.py          # Member 1
│   │   └── split_federated.py  # Member 1
│   ├── models/
│   │   ├── cnn_model.py        # Member 2
│   │   └── train_centralized.py # Member 2
│   ├── federated/
│   │   ├── server.py           # Member 2
│   │   ├── client.py           # Member 2
│   │   └── client_dp.py        # Member 2 (Stage 2)
│   ├── privacy/
│   │   └── dp_training.py      # Member 2 (Stage 2)
│   ├── interpretability/
│   │   ├── gradcam.py          # Member 3 (Stage 2)
│   │   └── shap_explain.py     # Member 3 (Stage 2)
│   ├── robustness/
│   │   └── adversarial.py      # Member 3 (Stage 2)
│   ├── attacks/
│   │   └── data_poisoning.py   # Member 1 (Stage 2)
│   └── evaluation/
│       ├── metrics.py          # Member 3
│       └── evaluate.py         # Member 3
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Member 1
│   ├── 02_baseline_training.ipynb     # Member 2
│   ├── 03_federated_training.ipynb    # Member 2
│   ├── 04_evaluation.ipynb            # Member 3
│   ├── 05_dp_experiments.ipynb        # Member 2 (Stage 2)
│   └── 06_interpretability.ipynb      # Member 3 (Stage 2)
├── results/
│   ├── stage1/
│   └── stage2/
├── reports/
│   ├── stage1_report.pdf
│   └── final_report.pdf
└── presentations/
    ├── stage1_presentation.pptx
    └── final_presentation.pptx
```

---

# Checklist Summary

## Stage 1 Checklist

### Member 1 - Data Lead
- [ ] Download COVIDx CXR-4 dataset
- [ ] Complete EDA notebook
- [ ] Implement preprocessing pipeline
- [ ] Create 3 federated client splits (non-IID)
- [ ] Generate client_X_data.npz files
- [ ] Write dataset documentation
- [ ] Contribute to Stage 1 report (dataset section)

### Member 2 - Modeling Lead
- [ ] Research and select model architecture
- [ ] Implement CNN model class
- [ ] Train centralized baseline
- [ ] Implement Flower server
- [ ] Implement Flower client
- [ ] Run federated experiments (3 clients)
- [ ] Save all model weights
- [ ] Contribute to Stage 1 report (model + FL section)

### Member 3 - Evaluation Lead
- [ ] Define all evaluation metrics
- [ ] Implement evaluation framework
- [ ] Evaluate centralized model
- [ ] Evaluate federated model
- [ ] Create comparison tables
- [ ] Generate visualizations (ROC, confusion matrix)
- [ ] Contribute to Stage 1 report (results section)
- [ ] Prepare Stage 1 presentation

## Stage 2 Checklist

### Member 1 - Data Lead
- [ ] Create 3 data variants for cross-evaluation
- [ ] Implement label flipping attack
- [ ] Implement backdoor attack
- [ ] Run cross-evaluation with all models
- [ ] Complete data card
- [ ] Contribute to final report (6 pages)

### Member 2 - Privacy Lead
- [ ] Implement DP-SGD with Opacus
- [ ] Train models with epsilon = 1, 5, 10
- [ ] Integrate DP with Flower client
- [ ] Privacy-utility trade-off analysis
- [ ] Cross-evaluate DP models
- [ ] Contribute to final report (6 pages)

### Member 3 - Interpretability Lead
- [ ] Implement Grad-CAM
- [ ] Implement SHAP explanations
- [ ] Test robustness against attacks
- [ ] Calibration analysis
- [ ] Final cross-model evaluation
- [ ] Contribute to final report (6 pages)
- [ ] Prepare final presentation

---

# Quick Reference: Key Libraries

```python
# Core
pip install torch torchvision
pip install flwr  # Flower for FL
pip install opacus  # Differential Privacy

# Evaluation
pip install scikit-learn
pip install matplotlib seaborn

# Interpretability
pip install grad-cam  # or pytorch-grad-cam
pip install shap
pip install captum  # PyTorch interpretability

# Data
pip install pandas numpy
pip install Pillow
pip install pydicom  # If DICOM images
```
