#!/usr/bin/env bash
# ==============================================================================
# install_OPAL-CSP.sh - Reproducible environment: OPAL-CSP
# Mirrors the "flow+eqv2" environment used to train the final OPAL-CSP
# checkpoints (same requirements.txt as the W&B runs).
# Self-contained: no external files required.
#
# USAGE:
#   chmod +x install_OPAL-CSP.sh
#   ./install_OPAL-CSP.sh
# ==============================================================================

set -euo pipefail

ENV_NAME="OPAL-CSP"
PYTHON_VERSION="3.11.9"

# OC20 repository required by EquiformerV2 (docs/env_setup.md)
OCP_REPO="https://github.com/FAIR-Chem/fairchem.git"
OCP_COMMIT="f83d150b5fcb940b814f25043fb00914945b8708"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${CYAN}======================================================${NC}"; \
          echo -e "${CYAN}  $*${NC}"; \
          echo -e "${CYAN}======================================================${NC}"; }

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

cat > "$WORK_DIR/constraints.txt" << 'CONSTRAINTS_EOF'
torch==2.3.1+cu121
torchvision==0.18.1+cu121
torchaudio==2.3.1+cu121
triton==2.3.1
numpy==1.26.4
sympy==1.13.1
CONSTRAINTS_EOF

# torch, torchvision, torchaudio, PyG CUDA wheels and ocp-models are installed in their own steps
cat > "$WORK_DIR/requirements.txt" << 'REQUIREMENTS_EOF'
absl-py==2.2.2
aiohappyeyeballs==2.6.1
aiohttp==3.11.16
aiosignal==1.3.2
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
ase==3.25.0
asttokens==3.0.0
attrs==25.3.0
bidict==0.23.1
certifi==2025.1.31
charset-normalizer==3.4.1
chemparse==0.3.2
click==8.1.8
cloudpickle==3.1.1
comm==0.2.2
contourpy==1.3.1
cycler==0.12.1
Dans_Diffraction==3.3.3
debugpy==1.8.14
decorator==5.2.1
dill==0.3.9
dnspython==2.7.0
docker-pycreds==0.4.0
e3nn==0.4.4
einops==0.8.1
executing==2.2.0
filelock==3.13.1
finetuning-scheduler==2.3.3
fonttools==4.57.0
frozenlist==1.5.0
fsspec==2024.6.1
gitdb==4.0.12
GitPython==3.1.44
grpcio==1.71.0
hydra-core==1.3.2
idna==3.10
importlib_metadata==8.6.1
ipykernel==6.29.5
ipython==9.1.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.6
jedi==0.19.2
Jinja2==3.1.4
joblib==1.4.2
jupyter_client==8.6.3
jupyter_core==5.7.2
jupyterlab_widgets==3.0.14
kiwisolver==1.4.8
latexcodec==3.0.0
lightning==2.3.2
lightning-utilities==0.14.3
llvmlite==0.44.0
lmdb==1.6.2
lxml==5.3.2
Markdown==3.8
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matminer==0.9.3
matplotlib==3.10.1
matplotlib-inline==0.1.7
mdurl==0.1.2
monty==2025.3.3
mpmath==1.3.0
multidict==6.4.3
multiprocess==0.70.17
narwhals==1.35.0
nest-asyncio==1.6.0
networkx==3.3
numba==0.61.2
numpy==1.26.4
omegaconf==2.3.0
opt_einsum==3.4.0
opt-einsum-fx==0.1.4
p_tqdm==1.4.2
packaging==24.2
palettable==3.3.3
pandarallel==1.6.5
pandas==2.2.3
parso==0.8.4
pastel==0.2.1
pathos==0.3.3
pexpect==4.9.0
pillow==11.0.0
platformdirs==4.3.7
plotly==6.0.1
poethepoet==0.10.0
pox==0.3.5
ppft==1.7.6.9
prompt_toolkit==3.0.50
propcache==0.3.1
protobuf==5.29.4
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
PyAstronomy==0.22.0
pybtex==0.24.0
pydantic==2.11.3
pydantic_core==2.33.1
Pygments==2.19.1
pymatgen==2025.4.10
pymongo==4.12.0
pyocse==0.1.2
pyparsing==3.2.3
python-dateutil==2.9.0.post0
python-dotenv==1.1.0
pytorch-lightning==2.5.1
pytz==2025.2
pyxtal==1.0.7
PyYAML==6.0.2
pyzmq==26.4.0
quantities==0.16.2
requests==2.32.3
rich==14.0.0
ruamel.yaml==0.18.10
ruamel.yaml.clib==0.2.12
scikit-learn==1.6.1
scipy==1.15.2
sentry-sdk==2.26.0
setproctitle==1.3.5
setuptools==75.8.0
six==1.17.0
SMACT==3.1.0
smmap==5.0.2
spglib==2.6.0
stack-data==0.6.3
submitit==1.5.2
sympy==1.13.1
tabulate==0.9.0
tensorboard==2.19.0
tensorboard-data-server==0.7.2
threadpoolctl==3.6.0
timm==0.4.12
toml==0.10.2
tomlkit==0.13.2
torch_geometric==2.5.3
torchcde==0.2.5
torchdiffeq==0.2.5
torchdyn==1.0.6
torchmetrics==1.7.1
torchsde==0.2.6
tornado==6.4.2
tqdm==4.67.1
traitlets==5.14.3
trampoline==0.1.2
typing_extensions==4.12.2
typing-inspection==0.4.0
tzdata==2025.2
uncertainties==3.2.2
urllib3==2.4.0
vasprun-xml==1.0.4
wandb==0.19.9
wcwidth==0.2.13
Werkzeug==3.1.3
wheel==0.45.1
widgetsnbextension==4.0.14
xmltodict==0.14.2
yarl==1.19.0
zipp==3.21.0
REQUIREMENTS_EOF

_get_mamba_root() {
    micromamba info --json 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('root prefix',''))" \
        2>/dev/null || true
}

# ==============================================================================
step "Step 0 - Checking system prerequisites"
# ==============================================================================
command -v micromamba >/dev/null 2>&1 \
    || error "micromamba not found. Install from: https://mamba.readthedocs.io"
command -v git >/dev/null 2>&1 \
    || error "git not found. Install with: sudo apt install git"

info "micromamba : $(micromamba --version)"
info "git        : $(git --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
    info "GPU        : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found — make sure the NVIDIA driver is installed."
fi

# ==============================================================================
step "Step 1 - Creating environment '$ENV_NAME' (Python $PYTHON_VERSION)"
# ==============================================================================
if micromamba env list | grep -qE "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
    warn "Environment '$ENV_NAME' already exists. Skipping creation."
    warn "To reinstall: micromamba env remove -n $ENV_NAME -y"
else
    micromamba create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -c conda-forge -y
    info "Environment '$ENV_NAME' created."
fi

MAMBA_ROOT="$(_get_mamba_root)"
[[ -z "$MAMBA_ROOT" ]] && MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

ENV_PREFIX="$MAMBA_ROOT/envs/$ENV_NAME"
ENVPIP="$ENV_PREFIX/bin/pip"
ENVPYTHON="$ENV_PREFIX/bin/python"

[[ -x "$ENVPIP" ]]    || error "pip not found at: $ENVPIP"
[[ -x "$ENVPYTHON" ]] || error "python not found at: $ENVPYTHON"

info "python : $("$ENVPYTHON" --version)"
info "pip    : $("$ENVPIP" --version)"

# ==============================================================================
step "Step 2 - Installing PyTorch 2.3.1+cu121"
# ==============================================================================
"$ENVPIP" install \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

"$ENVPYTHON" -c "
import torch
assert torch.__version__ == '2.3.1+cu121', f'Unexpected version: {torch.__version__}'
print(f'  torch: {torch.__version__}  |  cuda.is_available(): {torch.cuda.is_available()}')
"

# ==============================================================================
step "Step 3 - Installing PyTorch Geometric CUDA extensions (torch 2.3 / cu121)"
# ==============================================================================
"$ENVPIP" install \
    pyg-lib==0.4.0+pt23cu121 \
    torch_scatter==2.1.2+pt23cu121 \
    torch_sparse==0.6.18+pt23cu121 \
    torch_cluster==1.6.3+pt23cu121 \
    torch_spline_conv==1.2.2+pt23cu121 \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# ==============================================================================
step "Step 4 - Installing main packages (requirements + constraints)"
# ==============================================================================
"$ENVPIP" install \
    -c "$WORK_DIR/constraints.txt" \
    -r "$WORK_DIR/requirements.txt"

# ==============================================================================
step "Step 5 - Installing OC20 (ocp-models) for EquiformerV2"
# ==============================================================================
# Kept inside the environment: the editable install points here
OCP_DIR="$ENV_PREFIX/src/fairchem"
info "Pinned commit: $OCP_COMMIT"
info "Location     : $OCP_DIR"

if [[ ! -d "$OCP_DIR/.git" ]]; then
    git clone "$OCP_REPO" "$OCP_DIR"
fi
git -C "$OCP_DIR" checkout -q "$OCP_COMMIT"

OCP_UTILS="$OCP_DIR/ocpmodels/common/utils.py"
[[ -f "$OCP_UTILS" ]] || error "utils.py not found at: $OCP_UTILS"

"$ENVPYTHON" - "$OCP_UTILS" << 'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
old = '    finally:\n        registry.register("imports_setup", True)\n'
new = ('    finally:\n        import nets\n        import oc20.trainer\n'
       '        registry.register("imports_setup", True)\n')
if new in text:
    print(f"  Already patched: {path}")
else:
    assert text.count(old) == 1, "Patch anchor not found in utils.py"
    with open(path, 'w') as f:
        f.write(text.replace(old, new))
    print(f"  Patched: {path}")
PYEOF

"$ENVPIP" install -e "$OCP_DIR" --no-deps

# ==============================================================================
step "Final environment verification"
# ==============================================================================
"$ENVPYTHON" - << 'PYEOF'
import importlib.metadata as im
import sys

print(f"\nPython : {sys.version}\n")
errors = []

import torch
print(f"{'torch':<22}: {torch.__version__}")
print(f"{'cuda.is_available':<22}: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"{'cuda version':<22}: {torch.version.cuda}")

for pkg in ["torch_geometric", "pyg-lib", "torch_scatter", "torch_sparse",
            "torch_cluster", "torch_spline_conv", "lightning", "e3nn",
            "pymatgen", "pyxtal", "PyAstronomy", "wandb", "numpy", "ocp-models"]:
    try:
        print(f"{pkg:<22}: {im.version(pkg)}")
    except Exception as e:
        errors.append(f"{pkg}: {e}")

try:
    from ocpmodels.common.registry import registry
    from ocpmodels.common.utils import conditional_grad
    from ocpmodels.models.base import BaseModel
    from ocpmodels.models.scn.sampling import CalcSpherePoints
    from ocpmodels.models.scn.smearing import GaussianSmearing
    print(f"{'ocpmodels imports':<22}: OK")
except Exception as e:
    errors.append(f"ocpmodels imports: {e}")

try:
    import ocpmodels.common.utils as u
    src = open(u.__file__).read()
    assert "import oc20.trainer" in src
    print(f"{'OC20 utils.py patch':<22}: OK")
except Exception as e:
    errors.append(f"OC20 utils.py patch: {e!r}")

try:
    x = torch.tensor([1.0]).cuda()
    print(f"{'GPU tensor':<22}: OK ({x.device})")
except Exception as e:
    errors.append(f"GPU tensor: {e}")

print()
if errors:
    print("WARNINGS / ERRORS:")
    for e in errors:
        print(f"  x {e}")
    sys.exit(1)
print("ALL CHECKS PASSED")
PYEOF

"$ENVPIP" check

echo ""
info "============================================================"
info "  Installation complete. Environment: $ENV_NAME"
info "  To activate: micromamba activate $ENV_NAME"
info "============================================================"
