# Conditional Molecule Generation using Diffusion Models

## Chapter 3: Dataset

### 3.1 ZINC250k Dataset

The initial dataset used for training is the ZINC250k dataset, which contains nearly 250,000 molecular compounds with drug-like properties. These compounds are extracted from the ZINC database, a widely used resource in chemical informatics and drug discovery, making it an ideal choice for training molecular generative models.

### 3.2 Property Extraction

We used ADMETlab 3.0 to extract the following properties from the selected dataset.

#### 3.2.1 Medicinal Chemistry

Medicinal chemistry involves the identification, synthesis, and development of new chemical entities suitable for therapeutic use. It also includes the study of existing drugs, their biological properties, and their quantitative structure-activity relationships (QSAR).

**Medicinal Chemistry Properties**:
- QED
- SA score
- GASA
- Fsp3
- MCE-18
- NP score
- Lipinski Rule
- Pfizer Rule
- GSK Rule
- Golden Triangle
- PAINS
- Alarm NMR Rule
- BMS Rule
- Chelating Rule
- Colloidal Aggregators
- FLuc Inhibitors
- Blue Fluorescence
- Green Fluorescence
- Reactive Compounds
- Promiscuous Compounds

#### 3.2.2 Distribution

Drug distribution refers to how a drug moves to and from various tissues in the body and the amount of drug present in these tissues.

**Distribution Properties**:
- PPB
- VDss
- BBB
- Fu
- OATP1B1 inhibitor
- OATP1B3 inhibitor
- BCRP inhibitor
- MRP1 inhibitor
- BSEP inhibitor

#### 3.2.3 Metabolism

Drug metabolism refers to how specialized enzymatic systems break down drugs, determining their duration and intensity of action.

**Metabolism Properties**:
- CYP1A2 inhibitor / substrate
- CYP2C19 inhibitor / substrate
- CYP2C9 inhibitor / substrate
- CYP2D6 inhibitor / substrate
- CYP3A4 inhibitor / substrate
- CYP2B6 inhibitor
- CYP2C8 inhibitor
- HLM Stability

### 3.3 Dataset Preparation

The ZINC250k dataset primarily consists of molecules composed of nine key elements: Carbon (C), Nitrogen (N), Oxygen (O), Fluorine (F), Phosphorus (P), Sulfur (S), Chlorine (Cl), Bromine (Br), and Iodine (I).

#### 3.3.1 Graph Representation and Bond Encoding

Molecular structures are represented as graphs, where atoms represent nodes and bonds represent edges. Bond types are encoded using a multi-channel adjacency matrix that maps single, double, and triple bonds into numerical values. RDKit is used to convert SMILES strings into molecular graphs, and PyTorch Geometric is used to store data objects containing node features, bond features, and adjacency matrices.

#### 3.3.2 Dataset Processing

- Load raw molecular data from CSV/PKL files.
- Convert SMILES to molecular graphs.
- Store processed data as PyTorch tensors.

#### 3.3.3 Dataset Splitting and Property Selection

- Training Set: 90%
- Testing Set: 10%

The architecture allows flexibility in selecting any subset of properties extracted from ADMETlab 3.0.

#### 3.3.4 Dataset Statistics

| Dataset   | Number of Molecules | Number of Nodes | Node Types | Edge Types |
|-----------|----------------------|------------------|-------------|-------------|
| ZINC250k  | 245,490              | 6–38             | 9           | 3           |

---

## Chapter 4: Methodology

### 4.1 Model Description

Our model is based on Conditional Diffusion over Discrete Graph Structures (CDGS) to generate molecular graphs satisfying given properties.

**Key Components**:
- **Atom/Bond Feature Encoding**: Nodes and edges are encoded with vector representations and dense multi-channel adjacency matrices.
- **Hybrid Message-Passing**: Local GINEConv layers are combined with global graph transformers.
- **Dual Property Conditioning**:
  - Cross-Attention Guidance:

    Attention(Q, K, V) = softmax(QKᵀ / √dk) V

    where Q = graph features, K, V = property embeddings

  - Property Predictor: An MLP head predicts 36 molecular properties.
- **Noise Prediction Module**: Joint prediction of noise at atom and bond levels aligned with target properties.

### 4.2 Training Objective

The goal is to conditionally generate molecular graphs satisfying desired properties using:

- **Noise Prediction Loss**:

  L_noise = ||εX − ε̂X||² + ||εA − ε̂A||²

- **Property Prediction Loss**:

  L_prop = ||P − P̂||²

- **Combined Loss**:

  L = L_noise + wp · L_prop

The model employs stochastic differential equations (SDEs) and uses an Exponential Moving Average (EMA) model and AdamW optimizer with gradient clipping.

**Forward Diffusion Process**:

Gt = (α(t)X₀ + σ(t)εX, α(t)A₀ + σ(t)εA)

### 4.3 The Framework

- **Data Preprocessing**: Convert SMILES to graphs, normalize features, quantize bond types, and normalize properties.
- **Diffusion Process**:
  - Forward: Adds noise using a VPSDE schedule.
  - Reverse: Uses a noise prediction module with DPM-Solvers and cross-attention to generate graphs conditionally.
- **Property Conditioning**: Cross-attention and MLP property predictors guide generation.
- **Training**: The model learns to predict both noise and properties.
- **Sampling & Generation**: Denoising from noise using learned guidance to generate valid molecular graphs.

### 4.4 The Architecture

| Component                   | Description |
|----------------------------|-------------|
| Time/Property Embedding    | Sinusoidal for time; MLP for properties |
| Hybrid Message Passing     | GINEConv + Transformer with cross-attention |
| Attention Mechanisms       | Focus on property-guided generation |
| Noise Prediction Module    | Predicts noise for atoms and bonds |
| Property Predictor         | MLP-based property prediction from graph features |
| EMA                        | Ensures training stability |

### 4.5 Algorithms Used

#### Algorithm 1: Optimizing CDGS

1. Sample t ∼ U(0,1], εX, εA ∼ N(0,I)
2. Compute noisy graph: Gt = (α(t)X₀ + σ(t)εX, α(t)A₀ + σ(t)εA)
3. Quantize adjacency: Ât = quantize(At)
4. Predict noise and properties: ε̂X, ε̂A, P̂ = model(Gt, Ât, t, P)
5. Compute loss: L = L_noise + wp · L_prop
6. Backpropagate and optimize model

#### Algorithm 2: Graph DPM-Solver

1. Compute step size: hi = λ(ti) − λ(ti−1)
2. Quantize At
3. Predict: ε̂X, ε̂A, P̂ = model(X, A, t)
4. Compute gradients of property loss
5. Adjust noise predictions with gradients
6. Update X and A using reverse SDE
7. Repeat for all time steps

---

## Chapter 5: Results and Evaluation

### 5.1 Evaluation Metrics Used

#### 5.1.1 Fréchet ChemNet Distance (FCD)

FCD evaluates the similarity between generated and real molecules. A lower score indicates higher similarity. Our model achieved:

**FCD Score**: 37.89313220237589

#### 5.1.2 Relative Root Mean Squared Error (RRMSE)

RRMSE is preferred over RMSE as it normalizes the error with respect to actual values, making it scale-independent. It is more suitable for evaluating molecular properties with small magnitudes.

We generated 5,000 molecules and compared their predicted properties against the ground truth using RRMSE.

### Property-Wise RRMSE Scores

| Property                  | RRMSE     | Property                  | RRMSE     |
|---------------------------|-----------|---------------------------|-----------|
| GASA                      | 0.936     | HIA                       | 148.1     |
| QED                       | 0.3796    | F20%                      | 0.8921    |
| SAscore                   | 0.7044    | F30%                      | 0.7115    |
| Fsp3                      | 1.577     | F50%                      | 0.6002    |
| MCE-18                    | 0.4864    | PPB                       | 0.3714    |
| NPscore                   | 0.8547    | VDss                      | 9.591     |
| Alarm NMR Rule            | 0.7462    | Fu                        | 2.573     |
| BMS Rule                  | 0.9008    | BBB                       | 6.201     |
| Chelating Rule            | 0.9985    | OATP1B1 inhibitor         | 0.6889    |
| PAINS                     | 0.9887    | OATP1B3 inhibitor         | 0.5861    |
| Lipinski Rule             | 0.014     | BCRP inhibitor            | 10.29     |
| Pfizer Rule               | 0.138     | MRP1 inhibitor            | 0.3457    |
| GSK Rule                  | 0.1278    | Colloidal Aggregators     | 0.6736    |
| Golden Triangle           | 0.059     | FLuc Inhibitors           | 0.6971    |
| Caco-2 Permeability       | 0.1991    | Blue Fluorescence         | 3.572     |
| MDCK Permeability         | 0.0564    | Green Fluorescence        | 13.99     |
| PAMPA                     | 0.6258    | Reactive Compounds        | 0.7665    |
| Pgp Inhibitor             | 443.6     | Pgp Substrate             | 6.838     |

