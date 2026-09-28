import os
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
import ase.io

from ase import Atoms
from ase.visualize.plot import plot_atoms

class COLLData(InMemoryDataset):
    # urls for downloading dataset
    raw_urls = {
        "train": "https://api.figshare.com/v2/file/download/25605734",
        "val": "https://api.figshare.com/v2/file/download/25605740",
        "test": "https://api.figshare.com/v2/file/download/25605737"
    }
    def __init__(self, split, dataset_path="coll_xyz"):
        self.root_dir = dataset_path
        self.split = split.lower()
        assert self.split in ["train", "val", "test"], f"'{split}' is not a valid split"
        super(COLLData, self).__init__(self.root_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def preprocess(self):
        # this is the important function to implement. It should return a list of Data objects
        # In this case, our data is stored in an .xyz file, which can be read by ASE (https://wiki.fysik.dtu.dk/ase/index.html)
        # We read this file, get a list of molecules, then go through the molecules and create a Data object of each of them
        mol_list = ase.io.read(os.path.join(self.root_dir, "raw", self.xyz_file_name), ":")
        data_list = []
        for mol in mol_list:
            target = torch.tensor(float(mol.info["atomization_energy"] / mol.positions.shape[0]))  # want energy per atom, hence dividing with number of atoms
            atomic_numbers = torch.tensor(mol.arrays["numbers"])
            pos = torch.tensor(mol.positions).float()
            cell = 100*torch.eye(3).view(1, 3, 3)  # a very large cubic cell, only for visualization purposes
            data_list.append(Data(atomic_numbers=atomic_numbers,
                                 y=target,
                                 pos=pos,
                                 cell=cell,
                                 natoms=atomic_numbers.shape[0],

                                 ))
        return data_list
        
    @property
    def xyz_file_name(self):
        # help function
        return f"coll_v1.2_AE_{self.split}.xyz"
    
    @property
    def raw_file_names(self):
        return [self.xyz_file_name]

    @property
    def processed_file_names(self):
        return [f'processed_data_{self.split}.pt']

    def download(self):
        url = self.raw_urls[self.split]
        download_url(url, self.raw_dir, filename=self.xyz_file_name)

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.preprocess()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def batch_to_atoms(batch):
    # helper function for converting a batch into Atoms objects that can be visualized
    n_systems = batch.cell.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    average_pos = torch.mean(batch.pos, dim=0)
    batch.pos = batch.pos + (torch.tensor([10, 10, 10]) - average_pos)
    positions = torch.split(batch.pos, natoms)
    cells = 1/5 * batch.cell
    energies = batch.y.tolist()

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            cell=cells[idx].cpu().detach().numpy(),
            pbc=[True, True, True],
        )
        atoms_objects.append(atoms)

    return atoms_objects

def get_coll_loaders(batch_size):
    # get train/val/test dataloaders with COLL data
    loaders_list = []
    for split in ["train", "val", "test"]:
        data = COLLData(split)
        loaders_list.append(DataLoader(data, batch_size=batch_size, shuffle=split=="train"))  # only shuffling train set
    return loaders_list
