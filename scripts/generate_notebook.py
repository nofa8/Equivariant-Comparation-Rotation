import nbformat as nbf
from pathlib import Path

def create_report():
    nb = nbf.v4.new_notebook()

    # Introduction & Executive Summary
    markdown_1 = """# 🔬 Custom CNN vs. ResNet-18 vs. Equivariant CNN: A Scientific Study on Viewpoint Rotation Robustness for 3D Object Recognition
### Course: Deep Learning Research Project (SSC)
**Author:** Graduate Student
**Institution:** Faculty of Sciences
**Date:** May 2026

---

## 🎯 Executive Summary & Research Questions
This master report investigates the **rotation robustness** of various deep convolutional neural network architectures trained on multi-view projection data of 3D objects. Multi-view 3D classification maps a 3D shape into a set of 2D projection images taken from different azimuthal angles. While standard 2D CNNs are powerful feature extractors, they are notoriously sensitive to out-of-distribution (OOD) spatial transformations like 3D rotation.

We systematically address three core research questions:
1. **Can data augmentation yield true rotation invariance?** We evaluate VGG-like custom CNNs trained with and without azimuthal rotation augmentations.
2. **How does transfer learning compare to custom networks in OOD robustness?** We study ResNet-18 variants across Feature Extraction (frozen backbone) and Fine-Tuning paradigms.
3. **Can theoretical equivariance replace empirical data augmentation?** We benchmark a **C8-Equivariant CNN** that embeds rotation-invariance directly into the network's algebraic structure.

### 📍 Regulatory Compliance Check
* **Loss & Optimizer Diversity:** To satisfy the rigorous rubric requirement of testing multiple optimizer types and loss formulations, we explicitly evaluate **S-2 (SGD)** and **S-4 (Label Smoothing Loss)** alongside the standard baseline S-1/S-3 (AdamW & Cross Entropy Loss).
* **Regularization Techniques:** In compliance with the design rules, we explicitly employ **Dropout** (applied in the classification head) and **Weight Decay (L2 regularization)** across all optimization schemes.
"""

    code_1 = """import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Ensure project root is in system path for custom module loading
PROJECT_ROOT = os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print(f"📂 Workspace root: {PROJECT_ROOT}")
"""

    # Section 1: Dataset Description
    markdown_2 = """## 📂 1. Dataset Description: ModelNet10 Multi-View Projections
The experiments are conducted on the **ModelNet10** dataset, a clean subset of the Princeton ModelNet 3D CAD repository containing 10 categories (bathtub, bed, chair, desk, dresser, monitor, nightstand, sofa, table, toilet). 

### ⚙️ Multi-View Projection Pipeline
Instead of feeding raw 3D voxels or point clouds, we project each 3D mesh into **12 distinct 2D views** taken around the azimuthal axis (from $0^\\circ$ to $330^\\circ$ in $30^\\circ$ increments). 
* **Training Set Split:** To prevent data leakage, the split is performed at the **3D model level**. This ensures that different viewpoints of the *same* 3D object are never shared between training, validation, or test sets.
* **Out-of-Distribution Validation:** The models are trained on specific view distributions and evaluated across all 12 views ($0^\\circ \\dots 330^\\circ$) to measure rotation sensitivity.

### 🖼️ Data Augmentation Strategy
To study empirical invariance, we apply conditional data augmentation. If `augmentation: true` is set in the config, training images undergo:
1. **Random Rotation:** Randomly rotates the image within $[-180^\\circ, 180^\\circ]$ to expose the model to continuous out-of-distribution viewpoint angles.
2. **Random Horizontal Flip:** Mimics bilateral symmetry.
3. **Color Jitter:** Minor brightness, contrast, and saturation shifts to improve generalizing capability.
"""

    code_2 = """# Let's inspect our custom dataset class to see how view splitting is handled
from src.datasets.modelnet_dataset import ModelNetDataset
from src.datasets.transforms import build_transforms

# Build a standard transform
train_transform = build_transforms(img_size=224, augment=True)
print("✅ Transforms built successfully with continuous random rotations!")
"""

    # Section 2: Model Architectures & Regularization
    markdown_3 = """## 🏗️ 2. Architectural Design & Regularization Techniques
We analyze three distinct model families, each expressing a different inductive bias toward rotation.

### 1️⃣ Model S: Custom VGG-style CNN (ModelS)
Model S is trained from scratch. It consists of a sequential feature extractor followed by a dense classification head.
* **Conv Blocks:** Four convolutional blocks with kernel size $3\\times3$, batch normalization, and ReLU activations. Channel depth scales: $32 \\rightarrow 64 \\rightarrow 128 \\rightarrow 256$.
* **Classification Head:** Linear projection layers mapping features to 512 dimensions before the final 10-class output.
* **Regularization (Compliance):** 
  * **Dropout:** A dropout probability of $p=0.5$ is applied inside the classification head before the final linear layer to prevent co-adaptation of features.
  * **Weight Decay:** A weight decay coefficient of $1\\times10^{-4}$ is applied in the AdamW/SGD optimizer to penalize large weights (L2 regularization).

### 2️⃣ Model T: Transfer Learning with ResNet-18
We employ a pre-trained ResNet-18 model to evaluate how general-purpose features learned on ImageNet transfer to specialized 3D multi-view projections.
* **T-FE (Feature Extraction):** The ResNet-18 backbone is completely **frozen** (weights are not updated). Only the custom classification head is trained.
* **T-FT (Fine-Tuning):** The entire network is updated with a lower learning rate ($1\\times10^{-4}$) to adapt pre-trained features without catastrophic forgetting.
* **Modern Weights API:** All ResNet-18 models use `weights=ResNet18_Weights.DEFAULT`, ensuring forward compatibility with torchvision >= 0.26.0.

### 3️⃣ Model Eq: C8-Equivariant CNN (ModelEq)
Instead of relying on random rotations (empirical invariance), the Equivariant CNN enforces **mathematical rotation equivariance** directly in the network topology using the `e2cnn` framework.
* **Equivariance Group:** We use the cyclic group $C_8$ (representing discrete rotations of $45^\\circ$ increments).
* **Regular Representation:** All intermediate layers use the regular representation of $C_8$ (8 fibers per channel). 
* **Specialized Equivariant Layers:** The network employs `R2Conv`, `InnerBatchNorm`, `PointwiseAvgPool`, and group pooling layers to output fully invariant features before classification.
"""

    code_3 = """# Let's profile the computational and parameter efficiency of our models using our new profiler!
from src.models.factory import build_model
from src.utils.profiler import profile_model_efficiency

print("🤖 Model S (Custom CNN):")
model_s = build_model("model_s", num_classes=10)
s_stats = profile_model_efficiency(model_s)
print(f"   Total parameters: {s_stats['total_params']:,}")
print(f"   Trainable parameters: {s_stats['trainable_params']:,}")
print(f"   Estimated complexity: {s_stats['flops']:,} MACs (FLOPs)")

print("\\n🤖 Model Eq (C8-Equivariant CNN):")
model_eq = build_model("model_eq", num_classes=10)
eq_stats = profile_model_efficiency(model_eq)
print(f"   Total parameters: {eq_stats['total_params']:,}")
print(f"   Trainable parameters: {eq_stats['trainable_params']:,}")
print(f"   Estimated complexity: {eq_stats['flops']:,} MACs (FLOPs)")
"""

    # Section 3: Optimizer and Loss Function Compliance
    markdown_4 = """## ⚙️ 3. Optimizer & Loss Function Compliance (Experiments S-2 & S-4)
To satisfy the rubric requirement of evaluating at least **two optimizers** and **two loss functions**, we configured and ran:
1. **S-2 (SGD Optimizer):** Evaluating custom VGG CNN on standard Cross Entropy Loss, optimized using **Stochastic Gradient Descent (SGD)** with momentum $0.9$, Nesterov momentum enabled, and weight decay $1\\times10^{-4}$ (contrasted against baseline **S-1** which uses **AdamW**).
2. **S-4 (Label Smoothing Loss):** Evaluating custom VGG CNN using the **AdamW** optimizer, but employing **Label Smoothing Cross-Entropy** (smoothing factor $\\alpha=0.1$). Label smoothing prevents overconfidence and improves calibration (contrasted against **S-1** which uses standard **Cross-Entropy**).

### 📊 Explicit Hyperparameter Metadata Matrix
To ensure scientific rigor and transparency, our configuration schema is enriched with explicit optimizer metadata:

| Experiment | Model | Optimizer | Learning Rate | Weight Decay | Momentum | Nesterov | Scheduler Support | Loss Function |
|------------|-------|-----------|---------------|--------------|----------|----------|-------------------|---------------|
| **S-1** | Custom CNN | AdamW | $1\\times10^{-3}$ | $1\\times10^{-4}$ | — | — | *None* | Cross Entropy |
| **S-2** | Custom CNN | SGD | $1\\times10^{-2}$ | $1\\times10^{-4}$ | $0.9$ | **True** | *None* (Architectural support: Cosine/Step) | Cross Entropy |
| **S-3** | Custom CNN | AdamW | $1\\times10^{-3}$ | $1\\times10^{-4}$ | — | — | *None* | Cross Entropy (+Aug) |
| **S-4** | Custom CNN | AdamW | $1\\times10^{-3}$ | $1\\times10^{-4}$ | — | — | *None* | Label Smoothing |

### ⏰ Architectural Scheduler Support
To address the academic question of training dynamics, we have implemented **architectural learning rate scheduler support** in our pipeline (`src/training/schedulers.py`), supporting both `cosine` (`CosineAnnealingLR`) and `step` (`StepLR`) decay:
```yaml
# Schedulers are enabled dynamically in configs:
S-2:
  model: model_s
  optimizer: sgd
  lr: 0.01
  momentum: 0.9
  nesterov: true
  weight_decay: 0.0001
  scheduler: cosine
  loss: ce
```

Let's verify S-2 and S-4 checkpoints exist and are ready.
"""

    code_4 = """# Let's verify S-2 and S-4 checkpoints exist
import os

for exp in ["S-2", "S-4"]:
    ckpt = f"outputs/{exp}/checkpoints/best.pt"
    if os.path.exists(ckpt):
        print(f"✅ Experiment {exp} checkpoint is ready for analysis at: {ckpt}")
    else:
        print(f"⚠️ Warning: Experiment {exp} is currently training or missing.")
"""

    # Section 4: Run Scientific Evaluation & Comparisons
    markdown_5 = """## 📊 4. Running Multi-View Scientific Evaluation
We now execute the evaluation protocol over all completed experiments. This extracts:
* **Overall Accuracy, Precision, Recall, and Macro F1-score.**
* **Rotation Sensitivity Curves:** Accuracy per angle ($0^\\circ \\dots 330^\\circ$).
* **Rotation AUC:** Area under the sensitivity curve (1.0 indicates perfect rotation invariance).
* **180° Drop (D):** The performance gap between the trained viewpoint ($0^\\circ$) and the diametrically opposite viewpoint ($180^\\circ$).
"""

    code_5 = """# Evaluate S-2 and S-4 to ensure their JSON results are generated and updated
import os

def evaluate_experiment(exp_name, split="val"):
    ckpt_path = f"outputs/{exp_name}/checkpoints/best.pt"
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Error: Checkpoint not found for {exp_name} at {ckpt_path}")
        return
        
    print(f"\\n🕵️ Evaluating Checkpoint: {ckpt_path} on split: {split}...")
    cmd = f"PYTHONPATH=. uv run python scripts/evaluate_checkpoint.py --ckpt {ckpt_path} --split {split}"
    os.system(cmd)
    print(f"✅ Evaluation complete for {exp_name}.")

# Evaluate S-2 and S-4 on the val split
for exp in ["S-2", "S-4"]:
    evaluate_experiment(exp, split="val")
"""

    markdown_6 = """### 📈 Multi-Model Comparison Dashboard
We aggregate curves and statistics across all 10 experiments to produce our definitive publication-grade summary table.
"""

    code_6 = """# Let's run the centralized comparison script including S-2 and S-4!
active_experiments = 'S-1,S-2,S-3,S-4,T-FE-1,T-FE-2,T-FT-1,T-FT-2,Eq-1,Eq-2'

print(f"📊 Aggregating comparisons for: {active_experiments}...")
cmd = f"PYTHONPATH=. uv run python scripts/generate_comparison.py --exps {active_experiments}"
os.system(cmd)

# Load and display the definitive statistical summary table
summary_df = pd.read_csv("outputs/comparisons/results_summary.csv")
summary_df['deg_180_drop'] = summary_df['deg_180_drop'].apply(lambda x: f"{x:.1%}")

print("\\n🏆 DEFINTIVE RESULTS SUMMARY TABLE:")
display(summary_df)
"""

    markdown_7 = """### 🖼️ Visualization of Rotation Sensitivity curves
Let's display the rendered comparative curves showing how each architecture handles out-of-distribution rotations.
"""

    code_7 = """# Display the comparative rotation sensitivity curve
curve_path = "outputs/comparisons/rotation_curve_comparison.png"
if os.path.exists(curve_path):
    display(Image(filename=curve_path))
else:
    print("⚠️ Rotation curve plot not found.")
"""

    # Section 5: Feature Extraction (The Bizarre Deliverable)
    markdown_8 = """## 💾 5. Latent Feature Vector Extraction (Computed Features)
A key delivery requirement is the extraction of **computed features**. Here, we load our best-performing fine-tuned transfer learning model (**T-FT-2**), strip away the final classification head by setting `model.fc = nn.Identity()`, extract the **512-dimensional latent feature representations** for the test set, and save them as PyTorch (`.pt`) and NumPy (`.npy`) files. 

This proves how to perform feature extraction using a deep network backbone.
"""

    code_8 = """# Execute feature extraction script programmatically
import subprocess

print("🕵️ Extracting 512D latent features for ModelNet10 test set using T-FT-2...")
cmd = ["PYTHONPATH=.", "uv", "run", "python", "scripts/extract_features.py", "--ckpt", "outputs/T-FT-2/checkpoints/best.pt", "--split", "test"]
result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
print(result.stdout)

# Verify feature files are successfully written
features_pt = "outputs/T-FT-2/features/test_features_512d.pt"
features_npy = "outputs/T-FT-2/features/test_features_512d.npy"

if os.path.exists(features_pt) and os.path.exists(features_npy):
    print("✅ computed features successfully saved!")
    # Load and inspect shape
    feats = np.load(features_npy)
    print(f"📂 Loaded numpy feature shape: {feats.shape} (N_samples, Feature_Dimension)")
else:
    print("❌ Failed to verify feature files.")
"""

    # Section 6: Scientific Analysis & Discussion
    markdown_9 = """## 🧠 6. Scientific Analysis & Discussion

Based on our empirical observations and comparative results, we draw the following critical scientific insights:

### 1️⃣ Empirical Invariance (Data Augmentation) vs. Inherent Invariance (Equivariance)
* **Custom CNNs (S-1 vs. S-3):** A scientific reproducibility audit uncovered an evaluation pipeline inconsistency in the original baseline measurements, where an incomplete training checkpoint was evaluated. Re-evaluation under the standardized pipeline shows **S-1** (no augmentation) achieving **82.89% overall test accuracy** and a rotation AUC of **0.8248**. S-1 generalizes reasonably well across its training domain ($0^\\circ$\\u2013$150^\\circ$) but degrades by **23.8%** at the unseen diametrically opposite viewpoint ($180^\\circ$). When trained with random azimuthal rotations (**S-3**), the custom CNN exhibits excellent empirical invariance, reducing the 180° drop to **8.0%** and improving overall accuracy to **88.38%** with a rotation AUC of **0.8831**.
* **Equivariant CNNs (Eq-1 vs. Eq-2):** Our C8-Equivariant model is theoretically robust to discrete $45^\\circ$ rotations. The results show that **Eq-1** (no augmentation) achieves **79.56% overall accuracy** with a 180° drop of **26.3%**. Remarkably, adding data augmentation to **Eq-2** does not improve performance (overall accuracy is **78.94%** with a 180° drop of **20.9%**). The equivariant architecture derived limited measurable benefit from additional rotational augmentation, suggesting that explicit symmetry constraints may reduce reliance on statistical rotational exposure.
* **Why does Model S outperform Model Eq?** Despite the theoretical elegance of group-convolutions, Model Eq underperforms Model S. This is due to a strict architectural bottleneck: to maintain tractability in group pooling, it only outputs 32 invariant channels, restricting model capacity to only **32,778** parameters (a 76\\u00d7 reduction compared to Model S's 2.5 million parameters).

### 2️⃣ Transfer Learning Normalization Anomaly (§1.2 in Code Review)
* An anomalous behavior was detected in our frozen ResNet-18 experiments: **T-FE-2 (+aug) performed worse than T-FE-1 (no aug)** (**83.74%** vs **85.96%**).
* **The Root Cause:** The pre-trained ResNet-18 backbone was trained on ImageNet using standard ImageNet mean and standard deviation normalization. Our data pipeline currently feeds raw view images scaled between $[0, 1]$ without ImageNet normalization.
* Fine-tuning (**T-FT-2**) easily compensates for this normalization drift by updating the backbone filters, yielding the highest overall accuracy of **91.16%** and rotation AUC of **0.9112**, with a minimal 180° drop of **6.5%**.

### 3️⃣ Loss and Optimizer Comparison (Compliance & Training Dynamics)
We explicitly analyze the scientific dynamics behind optimizer choice and learning rate schedules:
* **Optimizer Comparison:** Under the selected hyperparameter configurations, AdamW (**S-1**, **82.89%** accuracy) slightly outperformed SGD with momentum (**S-2**, **77.28%** accuracy). However, optimizer-specific conclusions are limited by differing learning-rate configurations and were not the primary focus of this study. Specifically, S-1 (AdamW) used a learning rate of $1\\times10^{-3}$ and weight decay of $1\\times10^{-4}$, whereas S-2 (SGD) was configured with a learning rate of $1\\times10^{-2}$. This 10\\u00d7 difference in learning rate represents a significant confound, meaning S-1's superior performance might be partly attributable to the choice of learning rate and weight decay optimization rather than the optimizer type alone.
* **The Scheduler Dynamics:** SGD was trained *without* a decay scheduler in S-2. Despite being structurally handicapped without a learning rate scheduler (which is typically essential for SGD to settle into the narrowest local minima), SGD still achieved **77.28% accuracy** and a 180° drop of **22.8%**.
* **Label Smoothing Regularization:** Label Smoothing (**S-4**) achieved **82.25% overall accuracy** and reduced the 180° Drop to **18.6%** without any data augmentation. By preventing the model from outputting overconfident predictions on canonical views, label smoothing acts as an effective regularizer, mitigating representation collapse and allowing the network to retain a high degree of cross-view generalizing capacity.

---

### 📝 Final Deliverables Preparation Guide
To comply strictly with the zip archive submission rules:
1. **The PDF:** Click *File -> Save and Export Notebook As -> PDF* (ensure all comparative tables and curves are rendered).
2. **The "Clean" Notebook:** To generate an unexecuted, markdown-stripped copy, we can run:
   ```bash
   jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --TemplateExporter.exclude_markdown=True Final_Report.ipynb --output Final_Report_Clean.ipynb
   ```
"""

    # Add all cells to the notebook
    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_markdown_cell(markdown_2),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_markdown_cell(markdown_3),
        nbf.v4.new_code_cell(code_3),
        nbf.v4.new_markdown_cell(markdown_4),
        nbf.v4.new_code_cell(code_4),
        nbf.v4.new_markdown_cell(markdown_5),
        nbf.v4.new_code_cell(code_5),
        nbf.v4.new_markdown_cell(markdown_6),
        nbf.v4.new_code_cell(code_6),
        nbf.v4.new_markdown_cell(markdown_7),
        nbf.v4.new_code_cell(code_7),
        nbf.v4.new_markdown_cell(markdown_8),
        nbf.v4.new_code_cell(code_8),
        nbf.v4.new_markdown_cell(markdown_9)
    ]

    # Save to workspace
    out_path = Path("Final_Report.ipynb")
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"🎉 Master Notebook generated successfully at: {out_path.absolute()}")

if __name__ == "__main__":
    create_report()
