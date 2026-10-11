# Third-Party Notices

OPAL-CSP includes code and data from the open-source projects listed below. Each component remains under its original license, whose full text, including the copyright notice, is in the [`licenses/`](licenses/) folder.

## Code

| Project | License | Copyright | Files in this repository |
| --- | --- | --- | --- |
| [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2) | MIT ([text](licenses/LICENSE-EquiformerV2)) | (c) 2023 Yi-Lun Liao | Copied from `nets/equiformer_v2/`: `activation.py`, `drop.py`, `edge_rot_mat.py`, `gaussian_rbf.py`, `input_block.py`, `Jd.pt`, `layer_norm.py`, `module_list.py`, `radial_function.py`, `so2_ops.py`, `so3.py`, `wigner.py`. Modified: `transformer_block_adanorm.py` (from `transformer_block.py`) and `equiformer_v2_cell_adanorm.py` (from `equiformer_v2_oc20.py`). All in `py_files/modules/equiformer/` |
| [Open Catalyst Project (OCP)](https://github.com/FAIR-Chem/fairchem) | MIT ([text](licenses/LICENSE-OCP)) | (c) Facebook, Inc. and its affiliates | `py_files/modules/equiformer/so3.py` (included in EquiformerV2) |
| [e3nn](https://github.com/e3nn/e3nn) | MIT ([text](licenses/LICENSE-e3nn)) | (c) 2020 The Regents of the University of California, through Lawrence Berkeley National Laboratory, EPFL, Free University of Berlin and Kostiantyn Lapchevskyi | `py_files/modules/equiformer/wigner.py` (functions from e3nn 0.4.0, included in EquiformerV2) |
| [CrystalFlow](https://github.com/ixsluo/CrystalFlow) | MIT ([text](licenses/LICENSE-CrystalFlow)) | (c) 2024 Xiaoshan Luo | Polar lattice utilities (`lattice_polar_decompose`, `lattice_polar_build`, `lattice_polar_build_torch`), `build_crystal` and `find_symm_map` in `py_files/utils/data_utils.py`; `lattice_polar_build_torch` in `equiformer_v2_cell_adanorm.py`; `lattice_polar_build_torch`, `get_reciprocal_lattice_torch`, `RecSinusoidsEmbedding`, `SinusoidsEmbedding`, `SinusoidalTimeEmbeddings` and `CSPLayer` (modified) in `py_files/modules/gnn/gnn_adanorm.py` |
| [DiffCSP](https://github.com/jiaor17/DiffCSP) | MIT ([text](licenses/LICENSE-DiffCSP)) | (c) 2023 Rui Jiao | `lattice_params_to_matrix_torch` and `lattice_matrix_to_params` in `py_files/utils/data_utils.py`; `lattice_params_to_matrix_torch` and `SinusoidalTimeEmbeddings` in `equiformer_v2_cell_adanorm.py`. CrystalFlow is itself built on DiffCSP |
| [CDVAE](https://github.com/txie-93/cdvae) | MIT ([text](licenses/LICENSE-CDVAE)) | (c) 2021 Tian Xie, Xiang Fu | Lattice conversion utilities (`lattice_params_to_matrix_torch`, `lattice_matrix_to_params`, `abs_cap`), which DiffCSP inherits from CDVAE |

The OCP repository is not distributed with this code: `install_OPAL-CSP.sh` clones it from GitHub and patches it on the user's machine. Python packages installed with `pip` (PyTorch, PyTorch Geometric, e3nn, pymatgen, etc.) are used as libraries under their own licenses and are not redistributed.

## Data

The dataset splits in `data/` are identical to those distributed in the [DiffCSP](https://github.com/jiaor17/DiffCSP) repository (MIT, see above), which builds on the splits released with [CDVAE](https://github.com/txie-93/cdvae). The underlying structures come from:

| Dataset | Source | License |
| --- | --- | --- |
| `perov_5` | Castelli et al., *New cubic perovskites for one- and two-photon water splitting using the computational materials repository*, Energy & Environmental Science 5(10):9034–9043, 2012 | See the original publication |
| `carbon_24` | Pickard, *AIRSS data for carbon at 10 GPa and the C+N+H+O system at 1 GPa*, Materials Cloud, 2020 | CC BY 4.0 |
| `mp_20`, `mpts_52` | Jain et al., *The Materials Project: A materials genome approach to accelerating materials innovation*, APL Materials 1(1):011002, 2013 | CC BY 4.0 |

If you use these datasets, please cite the original sources above.
