# JEPA I-POMDP: Project State, Incident Recovery & Agent Handoff

**Handoff Date**: September 14, 2026  
**Repository**: `https://github.com/huytruong1810/jepa_ipomdp`  
**Current Branch**: `main` (commit `2686306` + checkpointer guard)  
**Test Suite Status**: **60 / 60 Passing** (`uv run pytest tests/ -v`, execution ~47s)  
**Hardware Baseline**: NVIDIA GeForce RTX 5080 Laptop GPU (16 GB VRAM, `sm_120`), Intel Core Ultra 9 275HX, Linux x86_64 (WSL2)

---

## 1. Incident Records: System Restarts & State Retrieval

### 1.1 Incident 1: Unexpected System Reboot (Sep 11, 2026)
* **Incident Timeline**:
  * **September 9, 2026, 21:45**: A 1,000,000-step training experiment on the Tiger domain was launched (`uv run main.py`) following complete resolution of theoretical defects P0–P4 (commit `2686306`).
  * **September 11, 2026, 06:29**: Host machine underwent an unexpected reboot (uptime log: `reboot system boot 6.18.33.2-micros Fri Sep 11 12:34 / 06:29`).
  * Continuous training ran uninterrupted for **~32.5 hours**, logging steps 0 through 141,500 before the shutdown.
  * **September 11, 2026, 12:50**: Session restored; agent resumed training from Step 141,500.

### 1.2 Incident 2: WSL Hang & Restart (Sep 14, 2026)
* **Incident Timeline**:
  * **September 11, 2026, 12:50**: Training resumed seamlessly from Step 141,500 (`uv run main.py`).
  * **September 14, 2026, 09:06**: Training ran continuously for **~68 hours** (reaching Step 383,475), when WSL became unresponsive, requiring a restart.
  * **September 14, 2026, 13:55**: Session restored; agent verified process/GPU state, validated checkpoint integrity, and resumed training from Step 383,000.

### 1.3 Preservation of Weights & Data (as of Sep 14, 2026)
No training progress or code changes were lost:
1. **Historical Best Model**: [`tiger_checkpoints/best_model.pt`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_checkpoints/best_model.pt) preserved at **Step 191,500** with loss **`0.00308`** (timestamp: Sep 12 00:31).
2. **Latest Checkpoint**: [`tiger_checkpoints/latest_checkpoint.pt`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_checkpoints/latest_checkpoint.pt) preserved at **Step 383,000** with loss **`0.39161`** (timestamp: Sep 14 08:54, ~12 min before hang).
3. **Telemetry Logs**: TensorBoard event streams in [`tiger_tensorboard/jepa_ipomdp_20260909-214505/`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_tensorboard/jepa_ipomdp_20260909-214505/) (122 MB) and [`tiger_tensorboard/jepa_ipomdp_20260911-125003/`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_tensorboard/jepa_ipomdp_20260911-125003/) (219 MB) remain fully intact.


---

## 2. Theoretical Remediation Prior to Interruption

As documented in [`BACKLOG.md`](file:///home/andyj1810/projects/jepa_ipomdp/BACKLOG.md), commit `2686306` resolved all critical theoretical defects:

* **T1 (VICReg Normalization)**: Normalized temporal VICReg variance and covariance penalties by the active sequence length (`vicreg_loss_accum / max(vicreg_steps, 1)`), eliminating the 10x gradient inflation.
* **T2 & T3 (Episodic Termination Discounting)**: Enforced $\gamma \cdot (1 - d_t)$ discounting across both multi-step dream rollouts and prioritized sequence buffer TD($\lambda$) returns to prevent value leakage across episode horizons.
* **T4 & T6 (Canonical Open-Loop MCTS & Dirichlet Noise)**: Fixed search to single latent observation mode (`num_latent_obs: 1`) and added Dirichlet exploration noise ($\alpha=0.3, \epsilon=0.25$) at root to break initial symmetry.
* **T5 (Zero-Initialized Distributional Heads)**: Initialized TwoHot value and reward projection layers to zero weights/biases to guarantee exact $0.0$ expected initial returns.
* **E1 & E2 (Exploration Annealing & Warmup)**: Integrated 100-episode buffer warmup and temperature annealing from 1.0 to 0.1.

---

## 3. Critical Fix Applied During Recovery: Checkpointer Guard

### Root Cause
During diagnosis of [`src/ipomdp/telemetry/checkpointer.py`](file:///home/andyj1810/projects/jepa_ipomdp/src/ipomdp/telemetry/checkpointer.py), an edge-case bug was identified:
* `ModelCheckpointer.__init__` initialized `self.best_loss = float('inf')`.
* When resuming from `tiger_checkpoints/latest_checkpoint.pt`, `self.best_loss` remained `float('inf')` because the filename did not contain `"best"`.
* Consequently, during subsequent periodic saves, any loss $< \infty$ (e.g. 0.20) would overwrite `best_model.pt`, clobbering the true historical minimum loss (`0.00567`).

### Solution Implemented
In [`src/ipomdp/telemetry/checkpointer.py`](file:///home/andyj1810/projects/jepa_ipomdp/src/ipomdp/telemetry/checkpointer.py):
```python
# Guard: Restore existing best_loss from best_model.pt if present on disk
best_path = self.save_dir / "best_model.pt"
if best_path.exists():
    try:
        best_ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        if 'loss' in best_ckpt and not torch.isinf(torch.tensor(best_ckpt['loss'])):
            self.best_loss = float(best_ckpt['loss'])
            self.logger.info(f"Initialized best_loss to {self.best_loss:.4f} from existing best_model.pt")
    except Exception as e:
        self.logger.warning(f"Failed to read existing best_loss from {best_path}: {e}")
```
This guarantees that whenever `main.py` is resumed, `best_model.pt` is strictly protected against regression.

### 3.2 CUDA Graph & Non-Finite Loss Safeguards (Sep 14, 2026)
* **CUDAGraph Mark Guard**: In [`main.py`](file:///home/andyj1810/projects/jepa_ipomdp/main.py), guarded `torch.compiler.cudagraph_mark_step_begin()` with `if use_compile and device.type == "cuda":` (preventing spurious calls to `currentStreamCaptureStatusMayInitCtx` when `compile: false`).
* **TwoHot Target Sanitization**: In [`src/ipomdp/models/distributions.py`](file:///home/andyj1810/projects/jepa_ipomdp/src/ipomdp/models/distributions.py), sanitized `targets_symlog` using `torch.nan_to_num(targets_symlog, nan=0.0, posinf=self.max_val, neginf=self.min_val)` before bin projection to eliminate NaN index overflow into scatter operations.
* **Loss Finiteness Check**: In [`src/ipomdp/training/trainer.py`](file:///home/andyj1810/projects/jepa_ipomdp/src/ipomdp/training/trainer.py), added an explicit `torch.isfinite(combined_loss)` guard before `combined_loss.backward()` to log and skip any corrupted updates, protecting CUDA memory state.

---

## 4. Current Execution Status

* **Status**: **Paused by User Request** (Clean Graceful Shutdown via SIGINT)
* **Paused At**: **Step 400,513 / 1,000,000** (September 14, 2026, 20:07:58)
* **Command to Resume**: `uv run main.py`
* **Telemetry & Artifacts**:
  * Checkpoints: [`tiger_checkpoints/latest_checkpoint.pt`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_checkpoints/latest_checkpoint.pt) (synced to **Step 400,513**, loss `0.57307`).
  * Interrupt Checkpoint: [`tiger_checkpoints/interrupt_checkpoint.pt`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_checkpoints/interrupt_checkpoint.pt) (saved at **Step 400,513**).
  * Best Model: [`tiger_checkpoints/best_model.pt`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_checkpoints/best_model.pt) (preserved at step 191,500 with loss `0.00308`).
  * Live TensorBoard: [`tiger_tensorboard/`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_tensorboard/)
  * Diagnostic Plots: [`tiger_plots/`](file:///home/andyj1810/projects/jepa_ipomdp/tiger_plots/)




---

## 5. Standard Operating Procedures (SOP) for Future Agents

If the machine reboots, crashes, or the session restarts again, future agents must follow these exact steps:

### Step 1: Verify Hardware & Process State
Check whether training is currently running:
```bash
pgrep -fl "main.py"
```
Check GPU status and VRAM allocation:
```bash
nvidia-smi
```

### Step 2: Check Checkpoint Timestamps & Steps
Inspect the step count and loss values without loading entire tensor state dicts:
```bash
uv run python -c "
import torch
for name in ['latest_checkpoint.pt', 'best_model.pt']:
    try:
        c = torch.load(f'tiger_checkpoints/{name}', map_location='cpu')
        print(f'{name}: Step {c.get(\"step\")} | Loss {c.get(\"loss\"):.5f}')
    except Exception as e:
        print(f'{name}: {e}')
"
```

### Step 3: Run Fast Verification Tests
Verify regression test suite passes before resuming:
```bash
uv run pytest tests/test_core_utils.py -v
```

### Step 4: Resume Training
To resume training seamlessly, execute:
```bash
uv run main.py
```
*Note*: `main.py` is configured by default to check `tiger_checkpoints/latest_checkpoint.pt`. It automatically loads all model weights, optimizer state, and step offsets, and safely initializes `best_loss` from `best_model.pt`.

### Step 5: Probing & Evaluation (Optional)
To evaluate checkpoints against Bayesian oracle bounds:
* **Representational Probing**:
  ```bash
  uv run probe_tiger.py --checkpoint tiger_checkpoints/best_model.pt
  ```
* **KL Divergence Evaluation**:
  ```bash
  uv run eval_kl_tiger.py --checkpoint tiger_checkpoints/best_model.pt
  ```
* **Visualizing Agent Gameplay**:
  ```bash
  uv run enjoy_tiger.py --checkpoint tiger_checkpoints/best_model.pt
  ```
