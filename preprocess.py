from common.utils import seed_everything, get_polymorphs, get_alignment_inputs, get_alignment_outputs
import warnings
warnings.filterwarnings('ignore')
from args import get_args
import glob
import os
import json

if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed)

    cif_paths = glob.glob(os.path.join(args.gt_path, args.dataset_name, '*.cif'))
    poly_path = os.path.join('poly_files')
    os.makedirs(poly_path, exist_ok=True)

    max_atoms = 52
    if 'mp_20' in args.dataset_name:
        max_atoms = 20
    elif 'perov_5' in args.dataset_name:
        max_atoms = 5
    elif 'carbon_24' in args.dataset_name:
        max_atoms = 24

    poly_file = os.path.join(poly_path, args.dataset_name + '.json')
    if os.path.exists(poly_file):
        formulas = json.load(open(poly_file))
    else:
        formulas = get_polymorphs(cif_paths)
        with open(poly_file, 'w') as f:
            json.dump(formulas, f, indent=2)

    annots_path = os.path.join(args.gt_path, args.dataset_name)
    predictions_path = os.path.join(args.pred_path, args.method_name, 'Cifs', args.dataset_name)
    inputs = get_alignment_inputs(formulas, annots_path, predictions_path,
                                  use_lcm=args.use_lcm, max_lcm=args.max_lcm)
    outputs = get_alignment_outputs(inputs, total_positions=None if args.use_lcm else max_atoms,
                                    batch_size=args.batch_size)

    output_dir = os.path.join('results', 'alignment', args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.method_name + '_alignment.json')
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=2)
