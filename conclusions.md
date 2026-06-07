# Rotation Robustness in 3D Object Classification: Comprehensive Analysis & Conclusions

This document synthesizes the empirical findings, architectural evaluations, and theoretical insights across all Jupyter notebooks in this project. It outlines the trade-offs between training models from scratch, transfer learning, and mathematically constrained equivariant architectures.

---

## 🎯 Executive Summary & Core Research Question

The central research question of this study is:
> *How do different deep learning strategies — training from scratch, transfer learning, and equivariant architectures — compare in terms of rotation robustness for 3D object classification from 2D multi-view renders?*

To answer this, the project evaluates three model families on multi-view projections (12 azimuthal angles from $0^\circ$ to $330^\circ$) of the **ModelNet10** dataset:
1. **Model S (Custom CNN):** VGG-style baseline trained from scratch.
2. **Model T (ResNet-18):** ImageNet-pretrained transfer learning baseline (Feature Extraction vs. Fine-Tuning).
3. **Model Eq (C8-Equivariant CNN):** `e2cnn`-based network incorporating discrete $C_8$ cyclic rotation symmetry.

---

## 📊 Section 1: Global Experimental Results & Comparison
### (`run_all_experiments.ipynb` & `Final_Report.ipynb`)

A centralized evaluation was performed across all 10 core experiment configurations. The test split contains all 12 views ($0^\circ \dots 330^\circ$) to measure out-of-distribution (OOD) generalization (note that models were trained only on 6 views: $0^\circ, 30^\circ, 60^\circ, 90^\circ, 120^\circ, 150^\circ$).

### Key Metrics Defined:
* **Rotation AUC:** Normalized Area-Under-the-Curve of test accuracy across all 12 views. A score of `1.0` represents perfect, view-independent classification.
* **180° Drop ($D$):** The accuracy gap between the canonical training viewpoint ($0^\circ$) and the diametrically opposite out-of-distribution viewpoint ($180^\circ$).

### Summary Results Table:
| Exp ID | Model Family | Strategy / Modifiers | Augmentation | Overall Test Acc | Macro F1 | Rotation AUC | 180° Drop ($D$) |
|--------|--------------|----------------------|:------------:|:----------------:|:--------:|:------------:|:---------------:|
| **S-1** | Custom CNN | AdamW (lr=1e-3) | ❌ | 82.89% | 0.7887 | 0.8248 | 23.8% |
| **S-2** | Custom CNN | SGD (lr=1e-2, no scheduler) | ❌ | 79.22% | 0.7459 | 0.7901 | 21.5% |
| **S-3** | Custom CNN | AdamW (lr=1e-3) + Aug | ✅ | 88.38% | 0.8347 | 0.8831 | 8.0% |
| **S-4** | Custom CNN | AdamW + Label Smoothing | ❌ | 84.10% | 0.7989 | 0.8363 | 18.7% |
| **T-FE-1**| ResNet-18 | Feature Extraction (Frozen) | ❌ | 85.96% | 0.8212 | 0.8586 | 18.4% |
| **T-FE-2**| ResNet-18 | Feature Extraction + Aug | ✅ | 83.74% | 0.7861 | 0.8372 | 12.6% |
| **T-FT-1**| ResNet-18 | Fine-Tuning (Full model) | ❌ | 88.86% | 0.8494 | 0.8880 | 9.8% |
| **T-FT-2**| ResNet-18 | Fine-Tuning + Aug | ✅ | **91.16%** | **0.8757**| **0.9112** | **6.5%** |
| **Eq-1** | Equivariant | C8-Equivariant CNN | ❌ | 79.56% | 0.7490 | 0.7942 | 26.3% |
| **Eq-2** | Equivariant | C8-Equivariant CNN + Aug | ✅ | 78.94% | 0.7140 | 0.7898 | 20.9% |

### Primary Conclusions:
1. **Rotational Augmentation works on standard CNNs:** Adding geometric augmentations to `Model S` (S-3) increases overall test accuracy by **5.49%** (absolute) and slashes the 180° performance drop from **23.8% to 8.0%**.
2. **Fine-Tuning is superior to Feature Extraction:** `T-FT-2` achieves the highest overall accuracy (**91.16%**) and the lowest rotation sensitivity ($D = 6.5\%$). Conversely, applying augmentation to a frozen backbone (`T-FE-2`) degrades performance, indicating that frozen ImageNet features cannot accommodate rotated synthetic CAD projections without weight adaptation.
3. **Equivariant CNNs are insensitive to augmentation:** The results for `Eq-1` (no aug) and `Eq-2` (with aug) are nearly identical (~79.5% vs ~78.9%). Because the equivariant convolution operator enforces rotational symmetry algebraically, exposing the network to rotated viewpoints during training yields no statistical performance gains.

---

## 🌀 Section 2: Theoretical & Pedagogical Insights into Equivariant CNNs
### (`understanding_e2cnn.ipynb`)

This notebook breaks down the mathematical foundations of Group Equivariant CNNs (G-CNNs) using the `e2cnn` library:

### 1. The Mathematical Failure Mode of Standard CNNs
Standard convolutions possess **translation equivariance** because sliding a kernel commute with spatial shifts:
$$[K * (T_t X)] = T_t [K * X]$$
However, standard convolutions **do not commute with rotations** ($R_\theta$):
$$[K * (R_\theta X)] \neq R_\theta [K * X]$$
Rotating the input image creates entirely different activation paths, causing downstream classifiers to fail on unseen views.

### 2. Steerable Kernels & Group Representations
Equivariant CNNs resolve this by constraining the convolution kernels to satisfy the **steerability constraint**:
$$K(g x) = \rho_{out}(g) K(x) \rho_{in}(g)^{-1} \quad \forall g \in G$$
Where $\rho(g)$ represents how a group element $g$ acts on the feature channels:
* **Trivial Representation:** $\rho(g) = 1$. The channel behaves as a scalar (invariant to rotations, e.g., input RGB intensity).
* **Regular Representation:** $\rho(g)$ is a permutation matrix. Rotating the input by $45^\circ$ **cyclically permutes the indices** of the channels.

### 3. Formal Equivariance Verification
By routing a rotated image through the model ($f(R_\theta x)$) and comparing it with the rotated output of the original image ($R_\theta f(x)$), the notebook numerically proves that:
* **Grid-Aligned Rotations ($90^\circ, 180^\circ, 270^\circ$):** Achieve perfect mathematical equivariance (Max Absolute Discrepancy $\approx 10^{-6}$ to $10^{-8}$).
* **Interpolated Rotations ($45^\circ$):** Exhibit minor numerical discrepancy ($\approx 0.1$) solely due to pixel-grid bilinear interpolation artifacts.

---

## 🏗️ Section 3: Equivariant Capacity & Bottleneck Exploration
### (`equivariant_architecture_exploration.ipynb`)

Despite having the theoretically correct inductive bias, the baseline equivariant model (`Eq-1`) underperformed the augmented custom CNN (`S-3`) by **8.82%**. This notebook explores this gap across three hypotheses: capacity constraints, group pooling compression, and enforced invariance.

### 1. Scaling Capacity Study
By expanding the number of fibers (channels) and depth in the equivariant layers, the capacity study yields dramatic improvements:

| Architecture Variant | Trainable Parameters | Overall Test Acc | Macro F1 | Acc / Million Params |
|----------------------|:--------------------:|:----------------:|:--------:|:--------------------:|
| **Eq-Small (Eq-1)** | 32,778 | 79.56% | 0.7490 | **2427.2** |
| **Eq-Medium** | 130,058 | 88.72% | 0.8439 | 682.2 |
| **Eq-Large** | 518,154 | 89.15% | 0.8493 | 172.1 |
| **Eq-Large-Deep** | 534,666 | **91.08%** | **0.8802** | 170.4 |
| **No-GPool (Ablation)**| 35,018 | **91.05%** | **0.8734** | **2600.1** |
| *Model S-3 (VGG Baseline)*| *2,492,170* | *88.38%* | *0.8347* | *35.5* |
| *Model T-FT-2 (ResNet-18)*| *11,444,298* | *91.16%* | *0.8757* | *8.0* |

### 2. Conclusions on Architectural Trade-offs:
1. **The performance gap is primarily a capacity issue:** Scaling parameters from **32k** (Eq-Small) to **534k** (Eq-Large-Deep) increases accuracy to **91.08%**, matching the fine-tuned ResNet-18 backbone while using **21× fewer parameters**.
2. **The GroupPooling Bottleneck is severe:** GroupPooling collapses representations to ensure rotation invariance (taking the max/average over the 8 cyclic representation channels). This results in an **8× information compression** (e.g., compressing 256 channels down to 32 channels before the classifier).
3. **Enforced invariance can be detrimental:** The **No-GPool** ablation skips group pooling, passing raw equivariant features directly to the classifier. Doing so increases Eq-Small's accuracy from **79.56% to 91.05%** (a 11.49% jump) with virtually no parameter increase.
   * *Why?* For 3D CAD classification, orientation is class-discriminative (e.g., an upside-down chair looks different from a table, whereas forcing perfect mathematical rotation invariance discards this vertical orientation cue).

---

## 🌀 Section 4: Group Order Exploration ($C_4$ vs. $C_8$ vs. $C_{16}$)
### (`group_order_exploration.ipynb`)

This notebook evaluates whether increasing the rotational resolution of the discrete symmetry group improves classification performance.

| Group | Order ($N$) | Step Size | Parameters | Test Accuracy | Macro F1 | Acc / Million Params | Compression Ratio |
|-------|:-----------:|:---------:|:----------:|:-------------:|:--------:|:--------------------:|:-----------------:|
| **C4** | 4 | $90^\circ$ | 16,650 | 86.95% | 0.8169 | **5222.2** | 4× |
| **C8** | 8 | $45^\circ$ | 32,778 | 79.56% | 0.7490 | 2427.2 | 8× |
| **C16**| 16 | $22.5^\circ$| 65,034 | **89.76%** | **0.8600** | 1380.2 | 16× |

### Key Findings:
1. **Non-linear Relationship:** Increasing the group order has two competing effects:
   * **Positive:** Finer rotational resolution approximates continuous rotation symmetry ($SO(2)$) more closely.
   * **Negative:** Larger groups cause more aggressive GroupPooling compression (C16 compresses channels by **16×**, whereas C4 only compresses by **4×**).
2. **Optimal Trade-off:** $C_{16}$ achieves the highest accuracy (**89.76%**) due to its high rotational resolution, but $C_4$ outperforms $C_8$ (**86.95%** vs. **79.56%**). This dip at $C_8$ indicates that at low model capacity, the 8× information loss outweighs the benefits of $45^\circ$ symmetry, whereas $C_4$ preserves more spatial descriptors (only 4× compression).

---

## ⚙️ Section 5: Empirical Hyperparameter Selection
### (`hyperparameter_selection.ipynb`)

To avoid arbitrary defaults, 5-epoch screening experiments were run to justify the choice of optimizer and loss function.

### 1. Optimizer Screening (CE Loss, C8-Equivariant CNN):
* **SGD (lr=1e-2, momentum=0.9, nesterov):** Best Val Acc = **58.20%**, Final Val Loss = 1.1453.
* **Adam (lr=1e-3):** Best Val Acc = **68.24%**, Final Val Loss = 0.8943.
* **AdamW (lr=1e-3):** Best Val Acc = **74.19%**, Final Val Loss = **0.7971**.
* *Conclusion:* **AdamW** converged significantly faster and achieved the lowest validation loss during early-stage training. It was thus selected for all primary runs.

### 2. Loss Function Screening (AdamW, C8-Equivariant CNN):
* **Cross-Entropy (CE):** Best Val Acc = **74.19%**, Final Train Loss = 0.8724.
* **Label Smoothing ($\epsilon=0.1$):** Best Val Acc = **71.41%**, Final Train Loss = 1.1886.
* *Conclusion:* Standard **Cross-Entropy** yielded superior accuracy and cleaner training signals on the balanced dataset without introducing additional hyperparameter tuning, leading to its selection.

---

## 🏁 Summary of Strategic Recommendations

Based on the collective notebook findings, we recommend the following design guidelines for rotation-robust image classification:

1. **If parameter count is unconstrained:** Fine-tune a high-capacity pretrained model (e.g. ResNet) with aggressive rotational data augmentation (`T-FT-2`). This yields the highest absolute classification accuracy (**91.16%**).
2. **If operating under strict parameter constraints (e.g., edge devices):** Use a Group Equivariant CNN without GroupPooling (`No-GPool`). It achieves **91.05%** accuracy with only **35k parameters**, offering a massive parameter efficiency boost (**Acc/M Params = 2600.1** vs. ResNet's **8.0**).
3. **Be cautious of GroupPooling:** Enforcing global mathematical invariance by pooling over group dimensions is mathematically elegant but drops valuable orientation cues. If the classification task contains orientation-sensitive classes, pass the equivariant representations directly to the classifier or scale up group order resolution ($C_{16}$).
