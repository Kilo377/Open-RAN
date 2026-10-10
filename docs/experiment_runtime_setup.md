# Experiment Runtime Setup

This note records the runtime assumptions for paper experiments.

## Sionna Device

The base PHY config is `configs/phy.yaml`. It uses CPU by default:

```yaml
sionna_phy:
  device: "cpu"
```

The paper intent-orchestration override `configs/phy_intent_orchestration.yaml`
uses the lightweight CPU PHY on native Windows:

```yaml
phy_type: "custom_phy"
sionna_phy:
  device: "cpu"
  gpu_strict: false
  gpu_memory_growth: true
```

This avoids requiring TensorFlow/Sionna in the base Python environment for the
full paper sweep. Sionna CPU/GPU remains available as a separate optional path,
but the default formal experiments now prioritize stable CPU execution.

The engine maps this field as follows:

- `device: "cpu"` -> TensorFlow device `/CPU:0`
- `device: "gpu"` -> TensorFlow device `/GPU:0`
- `device: "auto"` -> `/GPU:0` when TensorFlow sees a GPU, otherwise `/CPU:0`

To request GPU execution in another experiment, merge `configs/phy_gpu.yaml`
over the default PHY config or set the same field in the experiment-specific PHY
config:

```yaml
sionna_phy:
  device: "gpu"
```

This only works if the local TensorFlow installation can see a CUDA-capable GPU.
Before long runs, verify locally with:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means the run will not have a usable TensorFlow GPU device even if
the simulator config asks for `gpu`.

Current local check on this machine found an NVIDIA RTX 5060 through
`nvidia-smi`, but native Windows TensorFlow does not expose it as a GPU runtime.
We therefore use CPU for now. A temporary Windows conda environment named
`oran_sionna` was tested and then removed to keep the setup simple.

If a future GPU run is needed, recreate a separate environment from:

```powershell
C:\Users\Asus\miniconda3\Scripts\conda.exe env create -f environment-oran-sionna-windows.yml
C:\Users\Asus\miniconda3\Scripts\conda.exe env config vars set -n oran_sionna DRJIT_LIBLLVM_PATH="C:\Users\Asus\miniconda3\envs\oran_sionna\Library\bin\LLVM-C.dll"
```

For full GPU execution, install WSL2 from an elevated PowerShell window:

```powershell
.\scripts\install_wsl_for_sionna_gpu.ps1
```

The current Codex process is not elevated, so it cannot enable WSL system
features directly.

## LLM Provider

Paper intent-orchestration configs are now set up for a DeepSeek
OpenAI-compatible endpoint:

```yaml
llm_provider: "deepseek"
llm_api_key_env_var: "DEEPSEEK_API_KEY"
llm_base_url: "https://api.deepseek.com"
llm_model: "deepseek-v4-pro"
```

The runner preflight checks `DEEPSEEK_API_KEY` before launching full experiments.
Set it in PowerShell before running:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

The code still supports the generic `openai_compatible` and local `ollama`
providers, but the paper evaluation defaults now follow the requested DeepSeek
path.
