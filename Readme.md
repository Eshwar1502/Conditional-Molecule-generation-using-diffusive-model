# Conditional Molecule generation using diffusive model 

This repository implements **Conditional Molecule generation using diffusive model** for molecular graph generation. Our model generates molecular graphs conditioned on specific chemical and pharmacological properties such as QED, logP, solubility, and more.

---

## 🧠 Motivation

Traditional generative models like VAEs and GANs suffer from poor reconstruction quality and mode collapse. To overcome these limitations, we explore diffusion-based models that iteratively denoise molecular graphs while being guided by target molecular properties.

Our goal is to:
- Generate high-quality molecular structures under given conditions.
- Improve molecular diversity, validity, and novelty compared to existing methods.
- Accurately condition molecule generation on desired properties like QED, logP, solubility, etc.

---

## 📦 Features

- Conditional molecular generation based on 36 ADMET properties.
- Hybrid message passing blocks combining GINEConv and Transformer layers.
- Noise prediction model with property guidance.
- DPM-Solver-based sampling with configurable step size and batch size.
- Evaluation using Fréchet ChemNet Distance (FCD) and Relative Root Mean Squared Error (RRMSE).

---

## 📁 Dataset

We use the **ZINC250k dataset**, which contains ~245,490 drug-like molecules. Molecules are represented as graphs where atoms are nodes and bonds are edges.

### Property Extraction via ADMETlab 3.0

#### 🔬 Medicinal Chemistry

| Property                  | Description                                       |
|--------------------------|---------------------------------------------------|
| QED                      | Quantitative Estimate of Drug-likeness            |
| SA Score                 | Synthetic Accessibility Score                     |
| GASA                     | Ghose Filter Adjusted Surface Area                |
| Fsp³                     | Fraction of sp³-hybridized carbons                |
| MCE-18                   | Molecular Complexity Estimator                    |
| NP Score                 | Natural Product-likeness score                    |
| Lipinski Rule            | Rule-of-five compliance                           |
| Pfizer Rule              | Pfizer’s rule for drug likeness                   |
| GSK Rule                 | GlaxoSmithKline’s rule                            |
| Golden Triangle          | Rule for oral bioavailability                     |
| PAINS Alarm              | Pan Assay Interference Compounds                  |
| NMR Rule                 | Avoids compounds interfering in NMR assays        |
| BMS Rule                 | Bristol-Myers Squibb rule                         |
| Chelating Rule           | Avoids chelators                                  |
| Colloidal Aggregators    | Avoids aggregators                                |
| FLuc Inhibitors          | Firefly luciferase inhibitors                     |
| Blue/Green Fluorescence  | Fluorescent compounds                             |
| Reactive/Promiscuous     | Avoids non-specific reactivity                    |

#### 🧪 Distribution

| Property                | Description                                                |
|------------------------|------------------------------------------------------------|
| PPB                    | Plasma Protein Binding                                     |
| VDss                   | Volume of Distribution at Steady State                    |
| BBB                    | Blood-Brain Barrier Penetration                           |
| Fu                     | Fraction Unbound                                           |
| OATP1B1/1B3 Inhibitor  | Organic Anion Transporter Polypeptide Inhibitors          |
| BCRP Inhibitor         | Breast Cancer Resistance Protein Inhibitor                |
| MRP1 Inhibitor         | Multidrug Resistance-associated Protein Inhibitor         |
| BSEP Inhibitor         | Bile Salt Export Pump Inhibitor                           |

#### 🧬 Metabolism

| Property                        | Description                           |
|--------------------------------|---------------------------------------|
| CYP1A2–CYP3A4 Inhibitor/Substrate | Cytochrome P450 Enzyme Interactions |
| CYP2B6/CYP2C8 Inhibitor        | Additional Cytochrome Interactions    |
| HLM Stability                  | Human Liver Microsome Stability       |

### Dataset Statistics

| Metric              | Value                |
|---------------------|----------------------|
| Number of Molecules | 245,490              |
| Node Types          | 9 (C, N, O, F, P, S, Cl, Br, I) |
| Edge Types          | 3 (single, double, triple)      |
| Node Size Range     | 6–38 atoms           |
| Train/Test Split    | 90% / 10%            |

---

## ⚙️ Training

### Hidden Dimension = 256

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--config configs/vp_zinc_cdgs.py \
--mode train \
--workdir exp/vpsde_zinc_cdgs_256 \
--config.training.n_iters 2500000
```

### Hidden Dimension = 128

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--config configs/vp_zinc_cdgs.py \
--mode train \
--workdir exp/vpsde_zinc_cdgs_128 \
--config.training.batch_size 128 \
--config.training.eval_batch_size 128 \
--config.training.n_iters 2500000
```

Pretrained checkpoints are available in `exp/vpsde_qm9_cdgs`.

---

## 🔍 Sampling & Evaluation

Use DPM-Solver for conditional sampling:

### Example: Order 3, 50 Steps

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--config configs/vp_zinc_cdgs.py \
--mode eval \
--workdir exp/vpsde_zinc_cdgs_256 \
--config.eval.begin_ckpt 250 \
--config.eval.end_ckpt 250 \
--config.eval.batch_size 800 \
--config.sampling.method dpm3 \
--config.sampling.ode_step 50
```

### Additional Sampling Options

- Use `--config.eval.nspdk` for NSPDK evaluation.
- Adjust number of steps via `--config.model.num_scales YOUR_STEPS`.
- Control GPU memory usage via `--config.eval.batch_size`.

---

## 📊 Evaluation Metrics

### Fréchet ChemNet Distance (FCD)
Measures similarity between generated and real molecules. Lower values indicate better performance.  
**Result**: `FCD = 37.8931`

### Relative Root Mean Squared Error (RRMSE)
Normalized RMSE used to evaluate accuracy of predicted molecular properties.

| Property | RRMSE |
|----------|-------|
| GASA     | 0.936 |
| HIA      | 148.1 |
| QED      | 0.3796 |
| F20%     | 0.8921 |
| SAscore  | 0.7044 |
| F30%     | 0.7115 |
| Fsp3     | 1.577 |
| F50%     | 0.6002 |
| MCE-18   | 0.4864 |
| PPB      | 0.3714 |
| ...      | ...   |

*(See full table in the report)*

---

## 🏛️ Methodology

### Architecture Overview

| Component                  | Description                                              |
|---------------------------|----------------------------------------------------------|
| Time/Property Embedding   | Sinusoidal positional encoding + MLP embeddings          |
| Hybrid Message Passing    | 10 GINEConv layers + Transformer                         |
| Attention Mechanisms      | Cross-attention for property conditioning                |
| Noise Prediction Module   | Predicts noise in atom/bond features                     |
| Property Predictor        | MLP heads predicting molecular properties                |
| EMA                       | Exponential Moving Average for stable training           |

---

### Key Algorithms

#### Algorithm 1: Optimizing CDGS
1. Sample time and noise.
2. Corrupt input graph using schedule functions.
3. Quantize adjacency matrix.
4. Predict noise and properties.
5. Combine losses: `L_noise + w_p * L_property`.

#### Algorithm 2: Graph DPM-Solver
1. Start from noisy graph.
2. Iteratively denoise using DPM-Solvers.
3. Inject property gradients during sampling.
4. Post-process final graph to ensure validity.

---

