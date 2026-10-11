import argparse
import glob
import os
import shutil
import warnings
from collections import Counter

import numpy as np
from pymatgen.core import Structure
from tqdm import tqdm


def check_structure(cif_path: str, max_atoms: int | None = None) -> tuple[bool, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            struct = Structure.from_file(cif_path)
    except Exception as e:
        return False, f"parse error: {e}"

    if len(struct.sites) == 0:
        return False, "no atoms"

    if struct.lattice.volume <= 0:
        return False, f"non-positive volume ({struct.lattice.volume:.4f} Å³)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        det = np.linalg.det(struct.lattice.matrix)
    if not np.isfinite(det) or abs(det) < 1e-6:
        return False, f"singular cell matrix (det={det:.2e})"

    if max_atoms is not None and len(struct.sites) > max_atoms:
        return False, f"too many atoms ({len(struct.sites)} > {max_atoms})"

    return True, ""


def scan_directory(cif_dir: str, max_atoms: int | None, move: bool) -> list[tuple[str, str]]:
    cif_paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    if not cif_paths:
        return []

    invalid = []
    for path in tqdm(cif_paths, desc=f"  {os.path.relpath(cif_dir)}", leave=False):
        ok, reason = check_structure(path, max_atoms)
        if not ok:
            invalid.append((path, reason))
            if move:
                dest_dir = os.path.join(cif_dir, "invalid")
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(path, os.path.join(dest_dir, os.path.basename(path)))

    return invalid


def main():
    parser = argparse.ArgumentParser(description="Report or quarantine invalid CIF structures")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--gt_only", action="store_true")
    parser.add_argument("--pred_only", action="store_true")
    parser.add_argument("--max_atoms", type=int, default=None)
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--pred_path", default="pred_files")
    parser.add_argument("--gt_path", default="gt_files")
    args = parser.parse_args()

    dirs_to_scan = []

    if not args.pred_only:
        if args.dataset:
            gt_dir = os.path.join(args.gt_path, args.dataset)
            if os.path.isdir(gt_dir):
                dirs_to_scan.append(("gt", gt_dir))
        else:
            for d in sorted(os.listdir(args.gt_path)):
                gt_dir = os.path.join(args.gt_path, d)
                if os.path.isdir(gt_dir):
                    dirs_to_scan.append(("gt", gt_dir))

    if not args.gt_only:
        methods = [args.method] if args.method else sorted(os.listdir(args.pred_path))
        for method in methods:
            cifs_root = os.path.join(args.pred_path, method, "Cifs")
            if not os.path.isdir(cifs_root):
                continue
            datasets = [args.dataset] if args.dataset else sorted(os.listdir(cifs_root))
            for dataset in datasets:
                pred_dir = os.path.join(cifs_root, dataset)
                if os.path.isdir(pred_dir):
                    dirs_to_scan.append(("pred", pred_dir))

    if not dirs_to_scan:
        print("No directories found to scan.")
        return

    all_invalid = []
    for kind, cif_dir in dirs_to_scan:
        print(f"\n[{kind}] {cif_dir}")
        invalid = scan_directory(cif_dir, args.max_atoms, args.move)
        for path, reason in invalid:
            print(f"  INVALID  {os.path.basename(path)}: {reason}")
        all_invalid.extend(invalid)

    print(f"\n{'='*60}")
    print(f"Total invalid: {len(all_invalid)}")
    if all_invalid:
        reasons = Counter(r.split(":")[0] for _, r in all_invalid)
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")
        if args.move:
            print("Invalid CIFs moved to their respective 'invalid/' subfolders.")
        else:
            print("Run with --move to quarantine them automatically.")


if __name__ == "__main__":
    main()
