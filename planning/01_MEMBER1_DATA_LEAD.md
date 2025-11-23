# Member 1: Data & Experiment Design Lead
## Detailed Working Plan

---

## Role Summary

| Attribute | Details |
|-----------|---------|
| **Primary Role** | Data & Experiment Design Lead |
| **Main Responsibilities** | Dataset preparation, preprocessing, federated splits, cross-evaluation coordination |
| **Key Tools** | PyTorch, NumPy, Pandas, Matplotlib |
| **Trust Dimension (Stage 2)** | Data robustness, poisoning attacks |

---

## Stage 1: Baseline Development (Week 1-2)

### Task 1.1: Dataset Acquisition
**Duration:** Day 1-2
**Priority:** CRITICAL (blocks all other work)

#### Actionables:
1. **Download COVIDx CXR-4 dataset**
   - **Source**: https://www.kaggle.com/datasets/andyczhao/covidx-cxr2
   - **Alternative**: https://github.com/lindawangg/COVID-Net
   - **Download size**: ~31 GB (ZIP archive)
   - **Requires**: Kaggle account (free)

2. **Dataset Statistics (COVIDx CXR-4)**
   ```
   Total Images: 84,818
   Total Subjects: 45,342
   Format: ZIP archive with train/val/test splits
   ```

3. **Understand dataset structure**
   ```
   COVIDx-CXR4/
   ├── train.txt          # Training set labels/paths
   ├── val.txt            # Validation set labels/paths
   ├── test.txt           # Test set labels/paths
   └── images/            # All CXR images
       ├── image1.png
       ├── image2.png
       └── ...
   ```

4. **Label file format** (train.txt, val.txt, test.txt)
   ```
   # Each line: patient_id filename label datasource
   # Example: patient00001 image001.png positive cohen
   ```

5. **Classification variants to choose from**

   | Variant | Classes | Description |
   |---------|---------|-------------|
   | **Dataset A** | 3 classes | negative / non-COVID pneumonia / COVID-19 |
   | **Dataset B** | 2 classes | COVID-negative / COVID-positive |

   **Recommendation**: Start with Dataset B (binary) for simpler baseline, then optionally try Dataset A.

6. **Document metadata**
   - Total number of images per class
   - Image dimensions (variable - need resizing)
   - Patient-level grouping (important for splits!)
   - Train/val/test distribution

#### Deliverables:
- [ ] Downloaded dataset in `data/raw/`
- [ ] `data/README.md` with dataset statistics
- [ ] Decision on Dataset A vs B documented

#### Success Criteria:
- Dataset fully downloaded (~31GB) and extracted
- Can parse train.txt, val.txt, test.txt files
- Can load and display sample images
- Understand class distribution
- Decided on 2-class vs 3-class variant

---

### Task 1.2: Exploratory Data Analysis (EDA)
**Duration:** Day 2-3
**Priority:** HIGH

#### Actionables:
1. **Create Jupyter notebook** `notebooks/01_data_exploration.ipynb`

2. **Analyze class distribution**
   ```python
   # Example analysis for COVIDx CXR-4
   import pandas as pd
   import matplotlib.pyplot as plt

   def load_covidx_labels(txt_path):
       """Load COVIDx label file (train.txt, val.txt, test.txt)"""
       data = []
       with open(txt_path, 'r') as f:
           for line in f:
               parts = line.strip().split()
               if len(parts) >= 3:
                   patient_id, filename, label = parts[0], parts[1], parts[2]
                   data.append({'patient_id': patient_id, 'filename': filename, 'label': label})
       return pd.DataFrame(data)

   # Load all splits
   train_df = load_covidx_labels('data/raw/train.txt')
   val_df = load_covidx_labels('data/raw/val.txt')
   test_df = load_covidx_labels('data/raw/test.txt')

   print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

   # Class distribution
   print(train_df['label'].value_counts())

   # Visualize
   train_df['label'].value_counts().plot(kind='bar')
   plt.title('Training Set Class Distribution')
   plt.ylabel('Number of Images')
   plt.savefig('results/stage1/class_distribution.png')
   ```

3. **Analyze image properties**
   - Image sizes (height, width)
   - Pixel value distributions
   - Color channels (RGB vs grayscale)
   - File formats (PNG, JPEG, DICOM)

4. **Identify data quality issues**
   - Corrupted images
   - Duplicate images
   - Extreme outliers in pixel values
   - Inconsistent image sizes

5. **Visualize sample images**
   - Grid of samples from each class
   - Highlight visual differences between classes

#### Deliverables:
- [ ] `notebooks/01_data_exploration.ipynb` (complete EDA)
- [ ] `results/stage1/class_distribution.png`
- [ ] `results/stage1/sample_images.png`
- [ ] Summary statistics in notebook markdown

---

### Task 1.3: Preprocessing Pipeline
**Duration:** Day 3-4
**Priority:** HIGH

#### Actionables:
1. **Create preprocessing module** `src/data/preprocessing.py`

2. **Implement image transformations**
   ```python
   import torch
   from torchvision import transforms

   def get_train_transforms(img_size=224):
       return transforms.Compose([
           transforms.Resize((img_size, img_size)),
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.RandomRotation(degrees=15),
           transforms.ColorJitter(brightness=0.1, contrast=0.1),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust for grayscale
       ])

   def get_test_transforms(img_size=224):
       return transforms.Compose([
           transforms.Resize((img_size, img_size)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485], std=[0.229])
       ])
   ```

3. **Handle class imbalance**
   - Option A: Weighted sampling
   - Option B: Class weights in loss function
   - Option C: Data augmentation for minority classes
   - Document chosen approach with justification

4. **Create configuration file**
   ```yaml
   # configs/data_config.yaml
   preprocessing:
     image_size: 224
     normalize_mean: [0.485]
     normalize_std: [0.229]
     augmentation:
       horizontal_flip: 0.5
       rotation_degrees: 15
       brightness: 0.1
       contrast: 0.1
   ```

#### Deliverables:
- [ ] `src/data/preprocessing.py`
- [ ] `configs/data_config.yaml`
- [ ] Documentation of preprocessing decisions

---

### Task 1.4: PyTorch Dataset Class
**Duration:** Day 4-5
**Priority:** HIGH

#### Actionables:
1. **Create dataset module** `src/data/dataset.py`

2. **Implement COVIDxDataset class** (adapted for COVIDx CXR-4 structure)
   ```python
   import torch
   from torch.utils.data import Dataset
   from PIL import Image
   import os

   class COVIDxDataset(Dataset):
       """
       Dataset for COVIDx CXR-4.
       Reads from train.txt/val.txt/test.txt label files.
       """
       def __init__(self, data_dir, split='train', transform=None, binary=True):
           """
           Args:
               data_dir: Path to COVIDx data directory
               split: 'train', 'val', or 'test'
               transform: Torchvision transforms
               binary: If True, use 2-class (COVID+/COVID-). If False, use 3-class.
           """
           self.data_dir = data_dir
           self.split = split
           self.transform = transform
           self.binary = binary

           # Define class mapping
           if binary:
               # Dataset B: Binary classification
               self.classes = ['negative', 'positive']
           else:
               # Dataset A: 3-class classification
               self.classes = ['negative', 'pneumonia', 'COVID-19']

           self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
           self.samples = self._load_samples()

       def _load_samples(self):
           """Load samples from label txt file."""
           samples = []
           label_file = os.path.join(self.data_dir, f'{self.split}.txt')

           with open(label_file, 'r') as f:
               for line in f:
                   parts = line.strip().split()
                   if len(parts) >= 3:
                       patient_id, filename, label = parts[0], parts[1], parts[2]

                       # Map label to class index
                       if label in self.class_to_idx:
                           samples.append({
                               'patient_id': patient_id,
                               'path': os.path.join(self.data_dir, 'images', filename),
                               'label': self.class_to_idx[label]
                           })

           return samples

       def __len__(self):
           return len(self.samples)

       def __getitem__(self, idx):
           sample = self.samples[idx]
           image = Image.open(sample['path']).convert('RGB')
           label = sample['label']

           if self.transform:
               image = self.transform(image)

           return image, torch.tensor(label, dtype=torch.long)

       def get_patient_ids(self):
           """Return list of patient IDs (useful for patient-aware splits)."""
           return [s['patient_id'] for s in self.samples]
   ```

3. **Implement NPZ-based dataset for federated clients**
   ```python
   class COVIDxClientDataset(Dataset):
       """Dataset loading from .npz file for federated clients"""
       def __init__(self, npz_path, split='train', transform=None):
           data = np.load(npz_path)
           self.images = data[f'{split}_images']
           self.labels = data[f'{split}_labels']
           self.transform = transform

       def __len__(self):
           return len(self.images)

       def __getitem__(self, idx):
           image = self.images[idx]
           label = self.labels[idx]

           # Convert to PIL for transforms
           image = Image.fromarray(image)
           if self.transform:
               image = self.transform(image)

           return image, torch.tensor(label, dtype=torch.long)
   ```

4. **Test data loading**
   ```python
   # Test script
   dataset = COVIDxDataset('data/processed', split='train', transform=get_train_transforms())
   loader = DataLoader(dataset, batch_size=32, shuffle=True)

   for batch_images, batch_labels in loader:
       print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels.shape}")
       break
   ```

#### Deliverables:
- [ ] `src/data/dataset.py` with both dataset classes
- [ ] Working data loaders

---

### Task 1.5: Federated Data Splitting
**Duration:** Day 5-6
**Priority:** CRITICAL (blocks Member 2's FL work)

#### Actionables:
1. **Create splitting module** `src/data/split_federated.py`

2. **Implement non-IID splitting strategies**

   **IMPORTANT: Patient-Aware Splitting for Medical Data**
   ```python
   # COVIDx has patient_id for each image - MUST split by patient, not image!
   # Otherwise: data leakage (same patient in train AND test)

   def get_patient_groups(samples):
       """Group sample indices by patient_id."""
       from collections import defaultdict
       patient_to_indices = defaultdict(list)
       for idx, sample in enumerate(samples):
           patient_to_indices[sample['patient_id']].append(idx)
       return patient_to_indices
   ```

   **Strategy A: Label Skew (Dirichlet Distribution) - Patient-Aware**
   ```python
   import numpy as np
   from collections import defaultdict

   def dirichlet_split_patient_aware(samples, num_clients, alpha=0.5):
       """
       Split data using Dirichlet distribution for non-IID.
       PATIENT-AWARE: All images from same patient go to same client.
       Lower alpha = more non-IID (skewed)
       alpha=0.5 is commonly used in FL papers
       """
       # Group by patient first
       patient_to_indices = defaultdict(list)
       patient_to_label = {}

       for idx, sample in enumerate(samples):
           pid = sample['patient_id']
           patient_to_indices[pid].append(idx)
           patient_to_label[pid] = sample['label']  # Assume same patient = same label

       patients = list(patient_to_indices.keys())
       patient_labels = [patient_to_label[p] for p in patients]

       num_classes = len(np.unique(patient_labels))
       label_to_patients = defaultdict(list)

       for pid, label in zip(patients, patient_labels):
           label_to_patients[label].append(pid)

       client_patients = [[] for _ in range(num_clients)]

       # Distribute patients (not images) using Dirichlet
       for label in range(num_classes):
           pids = np.array(label_to_patients[label])
           np.random.shuffle(pids)

           proportions = np.random.dirichlet([alpha] * num_clients)
           proportions = (proportions * len(pids)).astype(int)
           proportions[-1] = len(pids) - sum(proportions[:-1])

           start = 0
           for client_id, num_patients in enumerate(proportions):
               client_patients[client_id].extend(pids[start:start + num_patients])
               start += num_patients

       # Convert patient lists back to image indices
       client_indices = []
       for client_pids in client_patients:
           indices = []
           for pid in client_pids:
               indices.extend(patient_to_indices[pid])
           client_indices.append(indices)

       return client_indices
   ```

   **Strategy B: Quantity Skew**
   ```python
   def quantity_skew_split(num_samples, num_clients, imbalance_factor=2.0):
       """
       Some clients have more data than others.
       imbalance_factor: ratio of largest to smallest client
       """
       # Generate skewed proportions
       proportions = np.random.lognormal(0, 0.5, num_clients)
       proportions = proportions / proportions.sum()

       # Ensure min samples per client
       min_samples = num_samples // (num_clients * 10)
       client_samples = (proportions * num_samples).astype(int)
       client_samples = np.maximum(client_samples, min_samples)

       return client_samples
   ```

3. **Create client data files**
   ```python
   def create_client_files(dataset, client_indices, output_dir, num_clients=3):
       """
       Save each client's data as NPZ file.
       """
       os.makedirs(output_dir, exist_ok=True)

       for client_id in range(num_clients):
           indices = client_indices[client_id]

           # Split into train/test (80/20)
           np.random.shuffle(indices)
           split_point = int(len(indices) * 0.8)

           train_indices = indices[:split_point]
           test_indices = indices[split_point:]

           # Extract images and labels
           train_images = np.array([dataset[i][0] for i in train_indices])
           train_labels = np.array([dataset[i][1] for i in train_indices])
           test_images = np.array([dataset[i][0] for i in test_indices])
           test_labels = np.array([dataset[i][1] for i in test_indices])

           # Save as NPZ
           np.savez(
               os.path.join(output_dir, f'client_{client_id + 1}_data.npz'),
               train_images=train_images,
               train_labels=train_labels,
               test_images=test_images,
               test_labels=test_labels
           )

           print(f"Client {client_id + 1}: {len(train_indices)} train, {len(test_indices)} test")
   ```

4. **Document data distribution per client**
   - Create visualization showing class distribution per client
   - Save statistics to CSV

#### Deliverables:
- [ ] `src/data/split_federated.py`
- [ ] `data/federated/client_1_data.npz`
- [ ] `data/federated/client_2_data.npz`
- [ ] `data/federated/client_3_data.npz`
- [ ] `results/stage1/client_distributions.png`
- [ ] `results/stage1/client_statistics.csv`

---

### Task 1.6: Project Infrastructure
**Duration:** Day 1 (parallel with download)
**Priority:** HIGH

#### Actionables:
1. **Create requirements.txt**
   ```
   # Core
   torch>=2.0.0
   torchvision>=0.15.0
   numpy>=1.24.0
   pandas>=2.0.0

   # Federated Learning
   flwr>=1.5.0

   # Visualization
   matplotlib>=3.7.0
   seaborn>=0.12.0

   # ML utilities
   scikit-learn>=1.3.0

   # Image processing
   Pillow>=10.0.0

   # Notebooks
   jupyter>=1.0.0
   ipykernel>=6.0.0

   # Privacy (Stage 2)
   opacus>=1.4.0

   # Interpretability (Stage 2)
   captum>=0.6.0
   shap>=0.42.0
   pytorch-grad-cam>=1.4.0
   ```

2. **Set up virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Or using conda
   conda create -n aitdm python=3.10
   conda activate aitdm
   pip install -r requirements.txt
   ```

3. **Create folder structure** (see PROJECT_OVERVIEW.md)

4. **Initialize Git properly**
   - Add `.gitignore`
   - Set up branch protection (if using GitHub)

#### Deliverables:
- [ ] `requirements.txt`
- [ ] `.gitignore`
- [ ] Complete folder structure
- [ ] Setup instructions in README

---

### Task 1.7: Stage 1 Report Contribution
**Duration:** Day 7 (Week 2 end)
**Priority:** HIGH

#### Sections to Write:
1. **Dataset Description** (1 page)
   - Dataset source and citation
   - Class descriptions
   - Data statistics (total samples, per-class counts)
   - Train/test split information

2. **Preprocessing Pipeline** (0.5 page)
   - Image transformations applied
   - Normalization strategy
   - Augmentation techniques
   - Class imbalance handling

3. **Federated Setup Description** (0.5 page)
   - Number of clients
   - Partitioning strategy (non-IID method used)
   - Data distribution per client
   - Visualization of client distributions

#### Deliverables:
- [ ] ~2 pages of Stage 1 report content

---

## Stage 2: Trustworthiness Enhancements (Week 3-5)

### Task 2.1: Create Data Variants for Cross-Evaluation
**Duration:** Week 3, Day 1-3
**Priority:** HIGH

#### Actionables:
1. **Design 3 distinct data variants**

   **Variant A: Different Label Distributions**
   - Client 1: Mostly COVID + Normal
   - Client 2: Mostly Lung_Opacity + Normal
   - Client 3: Balanced but smaller dataset

   **Variant B: Different Image Quality**
   - Client 1: High-resolution images
   - Client 2: Compressed/lower quality
   - Client 3: Mixed quality

   **Variant C: Temporal or Source Split** (if metadata available)
   - Split by image source/hospital
   - Split by collection date

2. **Implement variant creation scripts**
   ```python
   def create_label_skewed_variants(dataset, output_dir):
       """Create variants with different label distributions"""
       # Implementation
       pass
   ```

3. **Document each variant's characteristics**

#### Deliverables:
- [ ] 3 additional data variant NPZ files
- [ ] Documentation of variant differences
- [ ] Visualization comparing variants

---

### Task 2.2: Implement Data Poisoning Attacks
**Duration:** Week 3-4
**Priority:** MEDIUM

#### Actionables:
1. **Create attack module** `src/attacks/data_poisoning.py`

2. **Implement Label Flipping Attack**
   ```python
   def label_flipping_attack(labels, flip_rate=0.2, target_class=None):
       """
       Flip labels to poison the dataset.

       Args:
           labels: Original labels
           flip_rate: Percentage of labels to flip
           target_class: If specified, flip TO this class. Otherwise random.
       """
       poisoned_labels = labels.copy()
       num_to_flip = int(len(labels) * flip_rate)
       flip_indices = np.random.choice(len(labels), num_to_flip, replace=False)

       num_classes = len(np.unique(labels))
       for idx in flip_indices:
           if target_class is not None:
               poisoned_labels[idx] = target_class
           else:
               # Random flip to different class
               current = poisoned_labels[idx]
               new_label = np.random.choice([c for c in range(num_classes) if c != current])
               poisoned_labels[idx] = new_label

       return poisoned_labels, flip_indices
   ```

3. **Implement Backdoor Attack**
   ```python
   def add_trigger_pattern(image, trigger_size=10, trigger_value=255):
       """
       Add a trigger pattern (small square) to image corner.
       """
       triggered_image = image.copy()
       triggered_image[-trigger_size:, -trigger_size:] = trigger_value
       return triggered_image

   def backdoor_attack(images, labels, poison_rate=0.1, target_class=0):
       """
       Create backdoor: images with trigger are labeled as target_class.
       """
       num_to_poison = int(len(images) * poison_rate)
       poison_indices = np.random.choice(len(images), num_to_poison, replace=False)

       poisoned_images = images.copy()
       poisoned_labels = labels.copy()

       for idx in poison_indices:
           poisoned_images[idx] = add_trigger_pattern(images[idx])
           poisoned_labels[idx] = target_class

       return poisoned_images, poisoned_labels, poison_indices
   ```

4. **Create poisoned client datasets**
   - One clean client
   - One client with label flipping
   - One client with backdoor (for robustness testing)

#### Deliverables:
- [ ] `src/attacks/data_poisoning.py`
- [ ] Poisoned dataset variants
- [ ] Attack documentation

---

### Task 2.3: Cross-Evaluation Coordination
**Duration:** Week 4-5
**Priority:** HIGH

#### Actionables:
1. **Receive models from Member 2 and Member 3**
   - Centralized baseline
   - Federated model
   - DP models (epsilon = 1, 5, 10)

2. **Evaluate all models on your data variants**
   ```python
   def cross_evaluate(model_path, data_variant_path, metrics_fn):
       """
       Load model and evaluate on specific data variant.
       """
       model = load_model(model_path)
       dataset = load_dataset(data_variant_path)

       results = metrics_fn(model, dataset)
       return results
   ```

3. **Create cross-evaluation results table**

   | Model | Your Variant A | Your Variant B | Your Variant C |
   |-------|---------------|----------------|----------------|
   | Centralized | X% | X% | X% |
   | Federated | X% | X% | X% |
   | DP (ε=1) | X% | X% | X% |
   | DP (ε=5) | X% | X% | X% |
   | DP (ε=10) | X% | X% | X% |

4. **Analyze results**
   - Which models generalize best?
   - Which data variants are most challenging?
   - How does DP affect performance on different data?

#### Deliverables:
- [ ] Cross-evaluation results CSV
- [ ] Analysis notebook
- [ ] Visualizations comparing performance

---

### Task 2.4: Data Card and Documentation
**Duration:** Week 5
**Priority:** MEDIUM

#### Actionables:
1. **Create comprehensive data card** following ML Data Cards format

   ```markdown
   # COVIDx CXR-4 Data Card

   ## Dataset Overview
   - Name: COVIDx CXR-4
   - Version: 4.0
   - Task: Multi-class chest X-ray classification

   ## Intended Use
   - Primary: COVID-19 detection from chest X-rays
   - Research only, not for clinical deployment

   ## Data Composition
   - Total samples: X
   - Classes: COVID, Lung_Opacity, Normal, Viral_Pneumonia
   - Image format: PNG, grayscale/RGB

   ## Data Collection
   - Sources: [list hospitals/datasets]
   - Collection period: [dates]

   ## Ethical Considerations
   - Patient privacy: De-identified
   - Consent: [information]
   - Potential biases: [discuss]

   ## Known Limitations
   - Class imbalance
   - Geographic bias
   - Equipment variation
   ```

2. **Document all preprocessing decisions**

3. **Ethical considerations discussion**
   - Patient privacy in medical imaging
   - Consent and data sharing
   - Potential for harm from misclassification

#### Deliverables:
- [ ] `data/DATA_CARD.md`
- [ ] Ethical considerations section for report

---

### Task 2.5: Stage 2 Report Contribution
**Duration:** Week 5
**Priority:** HIGH

#### Sections to Write (~6 pages):
1. **Data Variants and Cross-Evaluation** (2 pages)
   - Description of 3 data variants
   - Justification for variant design
   - Cross-evaluation methodology
   - Results and analysis

2. **Data Poisoning Attacks** (2 pages)
   - Attack implementations
   - Attack parameters
   - Impact on model performance

3. **Ethical Considerations** (1 page)
   - Data privacy
   - Consent
   - Potential biases
   - Recommendations

4. **Data-related Future Work** (1 page)
   - Additional data sources
   - More sophisticated non-IID splits
   - Domain adaptation considerations

#### Deliverables:
- [ ] ~6 pages of final report content

---

## Communication Protocol

### With Member 2 (Modeling Lead):
- **Day 3-4:** Deliver preprocessed data samples for model testing
- **Day 5:** Deliver federated splits (NPZ files)
- **Week 4:** Receive DP models for cross-evaluation
- Share data statistics and class weights for loss function

### With Member 3 (Evaluation Lead):
- **Day 6:** Share data format documentation
- **Week 4:** Receive cross-evaluation results on your data
- **Week 5:** Coordinate on final comparison tables

### Sync Meetings to Attend:
- All team meetings (see PROJECT_OVERVIEW.md)
- Data handoff meeting with Member 2 (Day 4)

---

## Time Estimates

| Task | Estimated Hours | Priority |
|------|-----------------|----------|
| Dataset download | 2-4h | Critical |
| EDA notebook | 4-6h | High |
| Preprocessing pipeline | 3-4h | High |
| Dataset class implementation | 3-4h | High |
| Federated splitting | 4-6h | Critical |
| Project infrastructure | 2-3h | High |
| Stage 1 report sections | 4-6h | High |
| Data variants (Stage 2) | 4-6h | High |
| Poisoning attacks | 4-6h | Medium |
| Cross-evaluation | 6-8h | High |
| Data card | 2-3h | Medium |
| Stage 2 report sections | 8-10h | High |
| **Total** | **~50-60h** | |

---

## Checklist

### Stage 1
- [ ] Download COVIDx CXR-4 dataset
- [ ] Complete EDA notebook
- [ ] Implement preprocessing pipeline
- [ ] Create PyTorch dataset classes
- [ ] Create 3 federated client splits (non-IID)
- [ ] Generate client_X_data.npz files
- [ ] Set up project infrastructure
- [ ] Write Stage 1 report sections

### Stage 2
- [ ] Create 3 data variants for cross-evaluation
- [ ] Implement label flipping attack
- [ ] Implement backdoor attack
- [ ] Run cross-evaluation with all models
- [ ] Complete data card
- [ ] Write Stage 2 report sections
- [ ] Support team presentation preparation
