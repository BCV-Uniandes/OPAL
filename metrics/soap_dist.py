import numpy as np
from dscribe.descriptors import SOAP
from pymatgen.io.ase import AseAtomsAdaptor
import multiprocessing as mp


def _soap_worker(gt_structure, pred_structure, queue):
    try:
        adaptor = AseAtomsAdaptor()
        all_species = set()
        all_species.update([str(el) for el in gt_structure.composition.elements])
        all_species.update([str(el) for el in pred_structure.composition.elements])
        _soap = SOAP(species=list(all_species), r_cut=4.0, n_max=12, l_max=9,
                     sigma=0.5, periodic=True, average="outer")
        gt_fp = _soap.create(adaptor.get_atoms(gt_structure), n_jobs=1)
        pred_fp = _soap.create(adaptor.get_atoms(pred_structure), n_jobs=1)
        queue.put(float(np.linalg.norm(gt_fp - pred_fp)))
    except Exception:
        queue.put(None)


def soap(gt_structure, pred_structure):
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    p = ctx.Process(target=_soap_worker, args=(gt_structure, pred_structure, queue))
    p.start()
    p.join(timeout=120)
    if p.is_alive():
        p.terminate()
        p.join()
        raise RuntimeError("SOAP computation timed out")
    if p.exitcode != 0:
        raise RuntimeError(f"SOAP worker crashed (signal {-p.exitcode})")
    result = queue.get_nowait()
    if result is None:
        raise RuntimeError("SOAP computation failed")
    return result
