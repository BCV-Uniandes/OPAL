import os
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from py_files.data.dataset import CrystalDataset



class CrystalDataModule(pl.LightningDataModule):
    def __init__(self, data_path, num_workers, batch_size, num_atoms, use_space_group):
        super().__init__()
        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.use_space_group = use_space_group


    def setup(self, stage):
        train_paths = glob.glob(os.path.join(self.data_path,'train', '*.cif'))
        val_paths = glob.glob(os.path.join(self.data_path, 'val', '*.cif'))
        test_paths = glob.glob(os.path.join(self.data_path,'test', '*.cif'))
        data_name = os.path.split(self.data_path)[-1]
        c_train = os.path.join(self.data_path, 'train')
        c_val = os.path.join(self.data_path, 'val')
        c_test = os.path.join(self.data_path, 'test')
        csv_train = os.path.join(self.data_path, 'train.csv')
        csv_val = os.path.join(self.data_path, 'val.csv')
        csv_test = os.path.join(self.data_path, 'test.csv')
        self.train_ds = CrystalDataset(data_name +'_train', train_paths, c_train, csv_train, max_atoms=self.num_atoms, use_space_group=self.use_space_group)
        self.val_ds = CrystalDataset(data_name+'_val', val_paths, c_val, csv_val, max_atoms=self.num_atoms, use_space_group=self.use_space_group)
        self.test_ds = CrystalDataset(data_name+'_test', test_paths, c_test, csv_test, max_atoms=self.num_atoms, use_space_group=self.use_space_group)


    def train_dataloader(self):
        return DataLoader(self.train_ds, num_workers= self.num_workers, batch_size= self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers= self.num_workers, batch_size= self.batch_size, shuffle=False)
     
    def test_dataloader(self):
        return DataLoader(self.test_ds, num_workers= self.num_workers, batch_size= self.batch_size, shuffle=False)