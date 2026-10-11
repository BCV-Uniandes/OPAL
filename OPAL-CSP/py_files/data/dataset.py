import os
import torch
from torch.utils.data import Dataset
from py_files.utils.data_utils import preprocess_files, write_cifs_from_csvs




class CrystalDataset(Dataset):
    def __init__(self, name, cif_paths, save_path, csv_path = None, preprocess_workers = 30, max_atoms = 20, use_space_group = False):
        super().__init__()
        self.cif_paths = cif_paths
        self.csv_path = csv_path
        self.name = name
        self.preprocess(save_path, preprocess_workers, use_space_group)
        self.lattice_scaler = None
        self.scaler = None
        self.scalers = None
        self.max_atoms = max_atoms

    def preprocess(self, save_path, preprocess_workers, use_space_group):
        if use_space_group:
            save_path = save_path+'_sym'
        if os.path.exists(save_path+'.pt'):
            cached_data = torch.load(save_path+'.pt', weights_only=False)
        else:
            if len(self.cif_paths) == 0:
                split = self.name.split('_')[-1]
                print(f'There are not CIF files to process in {split}, creating cif files from csv...')
                self.cif_paths = write_cifs_from_csvs(self.csv_path, split)
            cached_data = preprocess_files(self.cif_paths, preprocess_workers, use_space_group)
            torch.save(cached_data, save_path+'.pt')
        self.cached_data = cached_data
    
    def get_raw_items(self):
        raw_coords = torch.zeros((self.max_atoms, 3))
        raw_atoms = torch.zeros((self.max_atoms))
        return raw_coords, raw_atoms


    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        spacegroup = data_dict['spacegroup']
        mp_id = data_dict['mp_id']
        coords, f_atoms = self.get_raw_items()
        frac_coords, atoms, lenghts, angles, num_atoms, lattice_polar = data_dict['crist_arrays']
        frac_coords = torch.tensor(frac_coords)
        atoms = torch.tensor(atoms)
        lenghts = torch.tensor(lenghts)
        angles = torch.tensor(angles)
        lattice_polar = torch.tensor(lattice_polar)
        coords[:len(frac_coords)] = frac_coords
        f_atoms[:len(atoms)] = atoms
        f_atoms = f_atoms.type(torch.int32)
        lattice_polar = lattice_polar.type(torch.float32)
        mask = f_atoms!=0
        mask = mask.unsqueeze(0).unsqueeze(0)
        return coords, f_atoms, mask, lenghts, angles, num_atoms, lattice_polar, mp_id, spacegroup