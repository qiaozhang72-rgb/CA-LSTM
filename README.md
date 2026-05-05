# CA-LSTM
This framework predicts transonic buffet flows via a POD-LSTM pipeline. Integrating Curriculum Learning and physics-inspired augmentation ensures stable, long-horizon autoregressive forecasting. It delivers a complete, data-driven ROM solution for high-fidelity, real-time unsteady flow prediction.

---

## 🔑 Core Features

| Module | Description |
|--------|-------------|
| **POD Data Normalization** (`guiyihua.py`) | Performs global Min-Max normalization of dimensional POD modal coefficients to `[-1, 1]`. Automatically saves channel-wise parameters for accurate physical quantity restoration. |
| **Sliding Window Restructuring** (`model_train.py`) | Converts temporal modal coefficients into a supervised learning tensor: `[samples, window_length, num_modes]`. |
| **Physics-Inspired Data Augmentation** | Supports temporally correlated noise (AR process), mode-dependent amplitude scaling, and amplitude/offset/mixed perturbations. Integrates a curriculum learning strategy for smooth transition scheduling. |
| **Recursive Prediction Monitoring** | Periodically runs multi-step autoregressive validation during training to track error accumulation. Dynamically checkpoints the best-performing weights. |
| **Autoregressive Inference Engine** (`model_ass_pre.py`) | Enables custom guide lengths for step-by-step rolling prediction. Includes built-in error statistics and multi-dimensional visual comparisons. |

---

## 🖥️ Environment Setup

```bash
Python >= 3.8
PyTorch >= 1.10
NumPy, Matplotlib, JSON, OS
```

> **Recommended:** Create an isolated environment using Anaconda:
> ```bash
> conda create -n lstm_pod python=3.9
> conda activate lstm_pod
> pip install torch numpy matplotlib
> ```

---

## 📂 Directory Structure

```
.
├── guiyihua.py          # Data normalization & parameter saving script
├── model_train.py       # Main training script (augmentation, training, recursive testing, visualization)
├── model_ass_pre.py     # Inference & autoregressive prediction script
├── dataSet/POD/         # Stores raw & normalized POD modal coefficients
├── model_trained/       # Auto-generated model weights directory
├── results_train/       # Training loss curves, config snapshots, training set predictions
└── results_asspre/      # Autoregressive predictions, error comparisons, detailed CSV logs
```

---

## 🚀 Quick Start

### 1. Data Normalization (`guiyihua.py`)
Normalizes raw dimensional POD modal coefficients and saves channel-wise parameters required for inverse transformation.

```bash
python guiyihua.py
```
**Outputs:**
- `normalized_modecoeff_M0.76_AOA6.dat`: Normalized coefficients (modes × time steps)
- `normalized_modecoeff_M0.76_AOA6.dat.json`: Per-mode `min`, `max`, and `range` values

> ⚠️ **Note:** Inference or physical quantity restoration must use the identical JSON parameter file.

### 2. Model Training (`model_train.py`)
After configuring hyperparameters, run the script. It automatically handles data loading, augmentation, training, recursive evaluation, and result saving.

```bash
python model_train.py
```
**Key Workflow:**
1. Loads normalized data and constructs sliding windows (default `yanchi=10`)
2. Applies augmentation (default `AUGMENT_RATIO=8`, curriculum learning enabled)
3. Trains LSTM; runs autoregressive error evaluation every `RECURSIVE_TEST_INTERVAL` epochs
4. Auto-saves optimal models, loss histories, and training prediction plots

### 3. Autoregressive Prediction (`model_ass_pre.py`)
Loads the trained model to perform long-horizon autoregressive prediction for specified flow conditions.

```bash
python model_ass_pre.py
```
**Configurable Parameters** (edit at the top of the script):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Mach`, `alpha` | Target flow condition (Mach number & AoA) | `'0.71'`, `'5.2'` |
| `yanchi` | Input window length | `20` |
| `pre_c_num` | Ground-truth guide length (warm-up steps) | `256` |
| `longtimepre` | Long-horizon mode (non-zero skips error calc, outputs only predictions) | `0` |
| `modelFilepath` / `modelFileName` | Model weight directory & filename | Auto-resolved |

---

## ⚙️ Core Parameters

### Data Augmentation (`model_train.py`)

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `AUGMENT_RATIO` | Augmentation multiplier | `4~8` |
| `NOISE_RANGE_MIN/MAX` | Noise perturbation amplitude | `0.01~0.05` |
| `CURRICULUM_LEARNING` | Enable curriculum learning | `True` |
| `TEMPORAL_CORRELATION` | Temporally correlated noise (AR(0.7)) | `True` |
| `MODE_DEPENDENT` | Adaptive noise scaling for higher-order modes | `True` |

### Training & Optimization

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `EPOCH` | Total training epochs | `1000~2000` |
| `LR` | Initial learning rate | `1e-3` |
| `hidden_num`, `num_layers` | LSTM hidden dimension & stack depth | `64~128`, `3~4` |
| `batch_size` | Mini-batch size | `64` |
| `RECURSIVE_TEST_INTERVAL` | Recursive validation interval (epochs) | `5~10` |

---

## 📄 Output Files

| Path | File | Description |
|------|------|-------------|
| `dataSet/POD/` | `normalized_*.dat` | Normalized modal coefficients (modes × time) |
| `dataSet/POD/` | `normalized_*.dat.json` | Normalization parameters (for inverse transformation) |
| `model_trained/{XLBB}/` | `*.pt01` | Best model via recursive testing |
| `model_trained/{XLBB}/` | `*.pt02` | Best model by training loss |
| `results_train/{XLBB}/` | `training_curves_*.png` | Training/validation loss curves |
| `results_train/{XLBB}/` | `config_summary_*.txt` | Full training configuration snapshot |
| `results_asspre/{XLBB}/` | `M*_AOA*_YC*_PCN*_e*.dat` | Autoregressive predictions (time, ground truth, prediction) |
| `results_asspre/{XLBB}/` | `prediction_comparison_*.png` | Multi-view comparison (line/scatter/error distribution) |
| `results_asspre/{XLBB}/` | `detailed_comparison_*.csv` | Step-wise, mode-wise detailed error metrics |

---

## 💡 Usage Tips & Notes

1. **Normalization Consistency:** Training and inference must share identical normalization parameters. Re-run `guiyihua.py` and update the JSON file if switching datasets.
2. **Window & Guide Length Alignment:** 
   - `yanchi` (input window) must match the training configuration.
   - `pre_c_num` (guide length) should be ≥ `yanchi`. Values between `100~512` significantly suppress error accumulation.
3. **Inverse Normalization Formula:**
   ```python
   # Load JSON parameters
   with open('params.json', 'r') as f:
       p = json.load(f)
   data_min, data_max = np.array(p['data_min']), np.array(p['data_max'])
   # Restore physical values
   original_pred = (normalized_pred + 1) / 2 * (data_max - data_min) + data_min
   ```
4. **Long-Horizon Prediction:** If `longtimepre != 0`, the script bypasses error computation and directly outputs the prediction sequence, ideal for generating continuous flow evolution data.
5. **Error Analysis:** Inspect error distribution histograms and scatter plots in `results_asspre/`. Systematic drift or deviation from `y=x` suggests increasing `pre_c_num` or incorporating a differential/gradient loss term.

---

##  License & Citation

This repository is provided for academic research purposes only. If used in publications, please cite relevant literature on flow field Reduced-Order Modeling (ROM) or LSTM-based time-series forecasting, and retain the repository link.

**Architectural References:**
- POD-based order reduction & modal extraction
- Curriculum learning & augmentation strategy: Implemented via the `PODNoiseAugmenter` class
- Autoregressive prediction framework: Sliding-window guidance + recursive rolling update

> **Applicable to:** CFD data-driven modeling, unsteady flow prediction, Reduced-Order Modeling (ROM), and deep learning in fluid dynamics.
