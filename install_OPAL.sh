#!/usr/bin/env bash
# ==============================================================================
# install_OPAL.sh - Reproducible environment: OPAL
# Self-contained: no external files required.
# Validated on: 2x Quadro RTX 8000, CUDA 12.4, Ubuntu 22.04, micromamba 2.5.0
#
# USAGE:
#   1. chmod +x install_OPAL.sh
#   2. ./install_OPAL.sh
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
ENV_NAME="OPAL"

# Exact commit from the validated environment — do not change
LEMAT_COMMIT="58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6"
LEMAT_REPO="https://github.com/LeMaterial/lemat-genbench.git"

# Exact commit from the validated environment — do not change
MATERIAL_HASHER_COMMIT="b3b33e4f8a57e6bc788c0c9facd3e6b55c153505"
MATERIAL_HASHER_REPO="https://github.com/lematerial/material-hasher.git"

# ------------------------------------------------------------------------------
# Colors and helpers
# ------------------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${CYAN}======================================================${NC}"; \
          echo -e "${CYAN}  $*${NC}"; \
          echo -e "${CYAN}======================================================${NC}"; }

# ------------------------------------------------------------------------------
# Temporary directory for embedded files (removed on exit)
# ------------------------------------------------------------------------------
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
info "Temporary directory: $WORK_DIR"

# ------------------------------------------------------------------------------
# Embedded constraints.txt
# ------------------------------------------------------------------------------
cat > "$WORK_DIR/constraints.txt" << 'CONSTRAINTS_EOF'
torch==2.6.0+cu124
torchvision==0.21.0+cu124
triton==3.2.0
sympy==1.13.1
numpy==2.3.5
setuptools==80.10.2
CONSTRAINTS_EOF

# ------------------------------------------------------------------------------
# Embedded requirements.txt (185+ packages)
# fairchem-core and mace-torch excluded: installed with --no-deps below
# ------------------------------------------------------------------------------
cat > "$WORK_DIR/requirements.txt" << 'REQUIREMENTS_EOF'
absl-py==2.4.0
aflow==0.0.11
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiohttp-cors==0.8.1
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.12.1
ase==3.27.0
ase_db_backends==0.11.0
attrs==25.4.0
average-minimum-distance==1.6.1
beautifulsoup4==4.14.3
bibtexparser==1.4.4
blake3==1.0.8
boto3==1.42.66
botocore==1.42.66
Brotli==1.2.0
cached_path==1.8.9
certifi==2026.2.25
cffi==2.0.0
charset-normalizer==3.4.5
click==8.3.1
cloudpickle==3.1.2
clusterscope==0.0.18
colorful==0.5.8
ConfigArgParse==1.7.5
contourpy==1.3.3
cryptography==46.0.5
cycler==0.12.1
Dans_Diffraction==3.4.0
datasets==4.7.0
dill==0.4.0
distlib==0.4.0
dm-tree==0.1.8
dnspython==2.8.0
dscribe==2.1.2
e3nn==0.5.0
emmet-core==0.86.3
fastapi==0.135.1
filelock==3.25.2
Flask==3.1.3
fonttools==4.62.0
frechetdist==0.6
frozenlist==1.8.0
fsspec==2026.2.0
func_timeout==4.3.5
gemmi==0.7.5
gitdb==4.0.12
GitPython==3.1.46
google-api-core==2.30.0
google-auth==2.49.0
google-cloud-core==2.5.0
google-cloud-storage==3.9.0
google-crc32c==1.8.0
google-resumable-media==2.8.0
googleapis-common-protos==1.73.0
grpcio==1.78.0
h11==0.16.0
h2==4.3.0
h5py==3.16.0
hf-xet==1.3.2
hpack==4.1.0
httpcore==1.0.9
httptools==0.7.1
httpx==0.28.1
huggingface_hub==1.6.0
hydra-core==1.3.2
hyperframe==6.1.0
idna==3.11
iniconfig==2.3.0
intel-cmplr-lib-ur==2025.3.2
intel-openmp==2025.3.2
itsdangerous==2.2.0
Jinja2==3.1.6
jmespath==1.1.0
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.0
latexcodec==2.0.1
lightning-utilities==0.15.3
llvmlite==0.46.0
lmdb==1.7.3
loguru==0.7.3
lxml==6.0.2
Markdown==3.10.2
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matminer==0.10.0
matplotlib==3.10.8
matscipy==1.2.0
mdurl==0.1.2
mkl==2025.3.1
mkl-service==2.6.1
monty==2026.2.18
moyopy==0.7.9
mp-api==0.46.0
mpmath==1.3.0
msgpack==1.1.2
multidict==6.7.1
multiprocess==0.70.18
munkres==1.1.4
narwhals==2.18.0
networkx==3.6.1
numba==0.64.0
numpy==2.3.5
omegaconf==2.3.0
onemkl-license==2025.3.1
opencensus==0.11.4
opencensus-context==0.1.3
opentelemetry-api==1.40.0
opentelemetry-exporter-prometheus==0.61b0
opentelemetry-proto==1.40.0
opentelemetry-sdk==1.40.0
opentelemetry-semantic-conventions==0.61b0
opt_einsum==3.4.0
opt-einsum-fx==0.1.4
orb-models==0.5.5
orjson==3.11.8
packaging==26.0
palettable==3.3.3
pandas==2.3.3
pathos==0.3.4
pillow==12.1.1
platformdirs==4.9.4
plotly==6.6.0
pluggy==1.6.0
point-cloud-utils==0.34.0
pox==0.3.7
ppft==1.7.8
prettytable==3.17.0
prometheus_client==0.24.1
propcache==0.4.1
proto-plus==1.27.1
protobuf==6.33.5
psutil==7.2.2
psycopg2-binary==2.9.11
p_tqdm==1.4.2
py-spy==0.4.1
pyarrow==23.0.1
pyasn1==0.6.2
pyasn1_modules==0.4.2
pybtex==0.25.1
pycparser==3.0
pydantic==2.12.5
pydantic_core==2.41.5
pydantic-settings==2.13.1
Pygments==2.19.2
pymatgen==2025.10.7
pymatgen-io-validation==0.1.2
pymongo==4.16.0
PyMySQL==1.1.2
pynndescent==0.6.0
pyparsing==3.3.2
pyre-extensions==0.0.32
PySocks==1.7.1
pytest==9.0.2
python-dateutil==2.9.0.post0
python-discovery==1.1.3
python-dotenv==1.2.2
python_hostlist==2.3.0
pytz==2026.1.post1
PyYAML==6.0.3
ray==2.54.0
referencing==0.37.0
requests==2.32.5
rich==13.9.4
rpds-py==0.30.0
rsa==4.9.1
ruamel.yaml==0.19.1
ruamel.yaml.clib==0.2.15
s3transfer==0.16.0
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
sentry-sdk==2.54.0
shellingham==1.5.4
six==1.17.0
SMACT==3.2.0
smart_open==7.5.1
soupsieve==2.8.3
sparse==0.18.0
spglib==2.7.0
starlette==0.52.1
structuregraph-helpers==0.0.9
submitit==1.5.4
sympy==1.13.1
tabulate==0.10.0
tbb==2022.3.1
tcmlib==1.4.1
tensorboard==2.20.0
tensorboard-data-server==0.7.2
termcolor==3.3.0
threadpoolctl==3.6.0
torch-ema==0.3
torch-geometric==2.7.0
torchmetrics==1.9.0
torchtnt==0.2.4
tqdm==4.67.3
typer==0.24.1
typing_extensions==4.15.0
typing-inspect==0.9.0
typing-inspection==0.4.2
tzdata==2025.3
umap-learn==0.5.11
umf==1.0.3
uncertainties==3.2.3
urllib3==2.6.3
uvicorn==0.41.0
uvloop==0.22.1
virtualenv==21.2.0
wandb==0.25.1
watchfiles==1.1.1
wcwidth==0.6.0
websockets==16.0
Werkzeug==3.1.8
wrapt==2.1.2
xxhash==3.6.0
yarl==1.23.0
zipp==3.23.1
REQUIREMENTS_EOF

# ------------------------------------------------------------------------------
# Detect micromamba root prefix
# ------------------------------------------------------------------------------
_get_mamba_root() {
    micromamba info --json 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('root prefix',''))" \
        2>/dev/null || true
}

# ==============================================================================
# Step 0 - Prerequisites check
# ==============================================================================
step "Step 0 - Checking system prerequisites"

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

if [[ "$LEMAT_COMMIT" == "main" ]]; then
    warn "LEMAT_COMMIT='main': the current HEAD of the main branch will be installed."
    warn "For full reproducibility, pin the commit by editing LEMAT_COMMIT at the top of this script."
    warn "See commits: https://github.com/LeMaterial/lemat-genbench/commits/main"
fi

# ==============================================================================
# Step 1 - Create base environment
# ==============================================================================
step "Step 1 - Creating environment '$ENV_NAME' (Python 3.11.15)"

if micromamba env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    warn "Environment '$ENV_NAME' already exists. Skipping creation."
    warn "To reinstall: micromamba env remove -n $ENV_NAME -y"
else
    micromamba create -n "$ENV_NAME" python=3.11.15 -c conda-forge -y
    info "Environment '$ENV_NAME' created."
fi

MAMBA_ROOT="$(_get_mamba_root)"
[[ -z "$MAMBA_ROOT" ]] && MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

ENVPIP="$MAMBA_ROOT/envs/$ENV_NAME/bin/pip"
ENVPYTHON="$MAMBA_ROOT/envs/$ENV_NAME/bin/python"

[[ -x "$ENVPIP" ]]    || error "pip not found at: $ENVPIP"
[[ -x "$ENVPYTHON" ]] || error "python not found at: $ENVPYTHON"

info "python : $("$ENVPYTHON" --version)"
info "pip    : $("$ENVPIP" --version)"

# ==============================================================================
# Step 2 - PyTorch 2.6.0 + CUDA 12.4
# ==============================================================================
step "Step 2 - Installing PyTorch 2.6.0+cu124"

"$ENVPIP" install \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

info "Verifying PyTorch + CUDA..."
"$ENVPYTHON" -c "
import torch
assert torch.__version__ == '2.6.0+cu124', f'Unexpected version: {torch.__version__}'
print(f'  torch: {torch.__version__}  |  cuda.is_available(): {torch.cuda.is_available()}')
"

# ==============================================================================
# Step 3 - PyTorch Geometric with CUDA
# ==============================================================================
step "Step 3 - Installing PyTorch Geometric (CUDA 12.4 / torch 2.6)"

"$ENVPIP" install \
    torch_scatter==2.1.2+pt26cu124 \
    torch_sparse==0.6.18+pt26cu124 \
    torch_cluster==1.6.3+pt26cu124 \
    torch_spline_conv==1.2.2+pt26cu124 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# ==============================================================================
# Step 4 - Pin numpy
# ==============================================================================
step "Step 4 - Installing numpy==2.3.5 (before remaining packages)"

"$ENVPIP" install "numpy==2.3.5"

# ==============================================================================
# Step 5 - Main packages with constraints
# ==============================================================================
step "Step 5 - Installing main packages (requirements + constraints)"

"$ENVPIP" install \
    -c "$WORK_DIR/constraints.txt" \
    -r "$WORK_DIR/requirements.txt"

# ==============================================================================
# Step 6 - fairchem-core without dependency resolution
# ==============================================================================
step "Step 6 - Installing fairchem-core==2.17.0 (--no-deps)"
# Declares torch~=2.8.0 but works with 2.6.0 (validated in original environment)

"$ENVPIP" install fairchem-core==2.17.0 --no-deps

# ==============================================================================
# Step 7 - mace-torch without dependency resolution
# ==============================================================================
step "Step 7 - Installing mace-torch==0.3.15 (--no-deps)"
# Declares e3nn==0.4.4 but the environment uses e3nn==0.5.0 (installed via fairchem)

"$ENVPIP" install mace-torch==0.3.15 --no-deps

# ==============================================================================
# Step 8 - Reinstall setuptools for pkg_resources
# ==============================================================================
step "Step 8 - Reinstalling setuptools==80.10.2 (required by torchtnt)"
# conda-forge setuptools >=81 dropped pkg_resources; PyPI 80.x includes it

"$ENVPIP" install --force-reinstall "setuptools==80.10.2"
"$ENVPYTHON" -c "import pkg_resources; print('  pkg_resources: OK')"

# ==============================================================================
# Step 9 - material-hasher from GitHub (exact commit)
# ==============================================================================
step "Step 9 - Installing material-hasher from GitHub"
info "Pinned commit: $MATERIAL_HASHER_COMMIT"

"$ENVPIP" install "git+${MATERIAL_HASHER_REPO}@${MATERIAL_HASHER_COMMIT}"

# ==============================================================================
# Step 10 - lemat-genbench from GitHub
# ==============================================================================
step "Step 10 - Installing lemat-genbench from GitHub (@${LEMAT_COMMIT})"

"$ENVPIP" install "git+${LEMAT_REPO}@${LEMAT_COMMIT}" --no-deps

# ==============================================================================
# Step 11 - Patch: rename uma-s-1 -> uma-s-1p2 in lemat_genbench
# ==============================================================================
step "Step 11 - Applying UMA model name patch (uma-s-1 -> uma-s-1p2)"
# fairchem-core 2.17.0 renamed the model; lemat_genbench still references the old name.
# The upstream lemat-genbench source already uses uma-s-1p2; this patch syncs
# the installed copy in the environment.

UMA_CALC="$MAMBA_ROOT/envs/$ENV_NAME/lib/python3.11/site-packages/lemat_genbench/models/uma/calculator.py"

[[ -f "$UMA_CALC" ]] || error "calculator.py not found at: $UMA_CALC"

"$ENVPYTHON" - "$UMA_CALC" << 'PYEOF'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    text = f.read()
text = text.replace('model_name: str = "uma-s-1"',         'model_name: str = "uma-s-1p2"')
text = text.replace('kwargs.pop("model_name", "uma-s-1")', 'kwargs.pop("model_name", "uma-s-1p2")')
text = text.replace('AVAILABLE_UMA_MODELS = ["uma-s-1"]',  'AVAILABLE_UMA_MODELS = ["uma-s-1p2"]')
text = text.replace('(default: "uma-s-1")',                 '(default: "uma-s-1p2")')
with open(path, 'w') as f:
    f.write(text)
print(f"  Patched: {path}")
PYEOF

info "Checking for remaining uma-s-1 references..."
if grep -qn '"uma-s-1"' "$UMA_CALC"; then
    warn "References to uma-s-1 still present — check manually:"
    grep -n '"uma-s-1"' "$UMA_CALC"
else
    info "Patch OK: no obsolete references remain"
fi

# ==============================================================================
# Final verification
# ==============================================================================
step "Final environment verification"

"$ENVPYTHON" - << 'PYEOF'
import importlib.metadata as im
import sys

print(f"\n=== System Info ===")
print(f"Python : {sys.version}")
errors = []

print(f"\n=== Verification ===")

import torch
print(f"{'torch':<22}: {torch.__version__}")
print(f"{'cuda.is_available':<22}: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"{'cuda version':<22}: {torch.version.cuda}")

for pkg in ["torch_scatter", "torch_sparse", "torch_cluster",
            "torch_spline_conv", "torch_geometric"]:
    try:
        print(f"{pkg:<22}: {im.version(pkg)}")
    except Exception as e:
        errors.append(f"{pkg}: {e}")

import pymatgen
print(f"{'pymatgen':<22}: {im.version('pymatgen')}")
import ase
print(f"{'ase':<22}: {ase.__version__}")

try:
    from fairchem.core import __version__ as fc_ver
    print(f"{'fairchem':<22}: {fc_ver}")
except Exception as e:
    errors.append(f"fairchem: {e}")

try:
    from mace import __version__ as mace_ver
    print(f"{'mace':<22}: {mace_ver}")
except Exception as e:
    errors.append(f"mace: {e}")

try:
    import pkg_resources
    print(f"{'pkg_resources':<22}: OK")
except Exception as e:
    errors.append(f"pkg_resources: {e}")

try:
    x = torch.tensor([1.0]).cuda()
    print(f"{'GPU tensor':<22}: OK ({x.device})")
except Exception as e:
    errors.append(f"GPU tensor: {e}")

for pkg in ["material-hasher", "lemat-genbench"]:
    try:
        print(f"{pkg:<22}: {im.version(pkg)}")
    except Exception as e:
        errors.append(f"{pkg}: {e}")

import importlib.util, pathlib
spec = importlib.util.find_spec("lemat_genbench")
if spec:
    calc = pathlib.Path(spec.origin).parent / "models" / "uma" / "calculator.py"
    if calc.exists():
        content = calc.read_text()
        if '"uma-s-1"' in content and '"uma-s-1p2"' not in content:
            errors.append("UMA patch NOT applied: calculator.py still uses uma-s-1")
        else:
            print(f"{'UMA patch':<22}: OK (uma-s-1p2)")

print()
if errors:
    print("WARNINGS / ERRORS:")
    for e in errors:
        print(f"  x {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
PYEOF

echo ""
info "============================================================"
info "  Installation complete. Environment: $ENV_NAME"
info "  To activate: micromamba activate $ENV_NAME"
info "============================================================"
