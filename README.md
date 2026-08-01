# PUF-ISACNet

**PUF-ISACNet** is a deep-learning framework for multi-class ocular disease classification from color fundus images. The project combines complementary convolutional and transformer-based feature extractors—**Inception-v3** and **Swin Transformer**—with attention-based feature fusion and a Transformer encoder to classify retinal images into ten diagnostic categories.

The implementation is provided as a Jupyter Notebook and is designed for GPU-enabled environments such as Kaggle.

## Project Description

PUF-ISACNet uses a hybrid architecture to capture both local pathological details and broader spatial relationships in fundus images:

- **Inception-v3** extracts multi-scale convolutional features.
- **Swin Transformer Tiny** captures hierarchical and long-range visual information.
- An **attention-based feature-fusion module** combines the two feature representations.
- A **two-layer Transformer encoder** refines the fused representation.
- A final linear classifier predicts one of ten ocular conditions.

The training pipeline includes class balancing through image augmentation, focal loss for difficult or underrepresented samples, center loss for more discriminative feature learning, progressive unfreezing of the Swin Transformer, gradient clipping, AdamW optimization, learning-rate warm-up, and cosine annealing.

## Supported Classes

The notebook classifies fundus images into the following ten categories:

1. CSCR Color Fundus
2. Diabetic Retinopathy
3. Disc Edema
4. Glaucoma
5. Healthy
6. Macular Scar
7. Myopia
8. Pterygium
9. Retinal Detachment
10. Retinitis Pigmentosa

## Model Architecture

```text
Input Fundus Image
        |
        +----------------------+
        |                      |
   Inception-v3         Swin Transformer Tiny
   2048-D features          768-D features
        |                      |
        +----------+-----------+
                   |
       Attention-Based Feature Fusion
                   |
             1024-D Features
                   |
       2-Layer Transformer Encoder
                   |
          Fully Connected Classifier
                   |
          10-Class Prediction
```

## Main Features

- Hybrid CNN–Transformer architecture
- Inception-v3 and Swin Transformer feature extraction
- Attention-guided feature fusion
- Transformer-based representation refinement
- Data augmentation and class balancing
- Focal loss with class weighting
- Center-loss regularization
- Progressive backbone unfreezing
- Multi-GPU support through `DataParallel`
- Classification report and confusion-matrix evaluation

## Reported Results

The stored notebook output reports the following validation performance:

| Metric | Score |
|---|---:|
| Accuracy | 91.68% |
| Macro Precision | 92.03% |
| Macro Recall | 91.62% |
| Macro F1-score | 91.76% |
| Weighted F1-score | 91.79% |

These values are based on the notebook's recorded validation run and may vary with the dataset split, random seed, hardware, package versions, and training configuration.

## Repository Structure

```text
PUF-ISACNet/
├── PUF-ISACNet.ipynb   # Data preparation, model training, and evaluation
├── README.md           # Project documentation
└── LICENSE             # MIT License
```

## Dataset

The notebook was developed using the **Original Eye Disease Image Dataset** available through Mendeley/Kaggle.

The expected directory structure is:

```text
Original Dataset/
├── CSCR_Color Fundus/
├── Diabetic Retinopathy/
├── Disc Edema/
├── Glaucoma/
├── Healthy/
├── Macular Scar/
├── Myopia/
├── Pterygium/
├── Retinal Detachment/
└── Retinitis Pigmentosa/
```

The dataset is not included in this repository. Download it separately and update the dataset path in the notebook when running outside Kaggle.

## Requirements

- Python 3.9 or later
- PyTorch
- torchvision
- timm
- transformers
- TensorFlow/Keras
- scikit-learn
- NumPy
- Matplotlib
- Seaborn
- tqdm
- Jupyter Notebook or JupyterLab

Install the main dependencies with:

```bash
pip install torch torchvision timm transformers tensorflow scikit-learn numpy matplotlib seaborn tqdm notebook
```

For GPU acceleration, install a PyTorch build compatible with your CUDA version.

## Installation

Clone the repository:

```bash
git clone https://github.com/kowshir-bitto/PUF-ISACNet.git
cd PUF-ISACNet
```

Install the dependencies:

```bash
pip install torch torchvision timm transformers tensorflow scikit-learn numpy matplotlib seaborn tqdm notebook
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `PUF-ISACNet.ipynb` and run the cells in sequence.

## Configuration

Before running the notebook, review and update these settings:

```python
data_dir = "/path/to/balanced/dataset"
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
desired_count = 1349
```

Also ensure that any pretrained-model checkpoint variable used by the image processor is defined. For the Swin Tiny backbone, a suitable Hugging Face checkpoint is:

```python
checkpoint = "microsoft/swin-tiny-patch4-window7-224"
```

## Training Workflow

1. Load the original fundus-image dataset.
2. Balance the classes using image augmentation.
3. Resize and normalize images.
4. Split the balanced dataset into training and validation sets.
5. initialize the Inception–Swin hybrid model.
6. Train using focal loss, center loss, AdamW, and learning-rate scheduling.
7. Progressively unfreeze Swin Transformer layers.
8. Evaluate the model using precision, recall, F1-score, accuracy, and a confusion matrix.

## Reproducibility Notes

The current notebook uses `random_split` without a fixed generator. To obtain repeatable splits, set random seeds before data preparation and pass a seeded generator:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

generator = torch.Generator().manual_seed(seed)
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)
```

## Limitations

- The notebook uses environment-specific Kaggle file paths that must be changed for local execution.
- The dataset is not bundled with the repository.
- The reported performance comes from a validation split rather than an external clinical test set.
- This software is intended for research and educational use and is not a substitute for professional medical diagnosis.
- Clinical deployment would require independent validation, calibration, regulatory review, and evaluation across diverse populations and imaging devices.

## Citation

A formal publication citation has not yet been added. When using this repository in academic work, cite the repository until a paper citation becomes available:

```bibtex
@software{bitto_puf_isacnet,
  author  = {Abu Kowshir Bitto},
  title   = {PUF-ISACNet: A Hybrid Inception-Swin Architecture for Ocular Disease Classification},
  url     = {https://github.com/kowshir-bitto/PUF-ISACNet},
  year    = {2025}
}
```

## Contributing

Contributions, bug reports, and improvement suggestions are welcome. Please open an issue or submit a pull request with a clear description of the proposed change.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Author

**Abu Kowshir Bitto**

GitHub: [@kowshir-bitto](https://github.com/kowshir-bitto)
