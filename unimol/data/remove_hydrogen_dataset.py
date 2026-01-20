# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


# class RemoveHydrogenDataset(BaseWrapperDataset):
#     def __init__(
#         self,
#         dataset,
#         atoms,
#         coordinates,
#         remove_hydrogen=False,
#         remove_polar_hydrogen=False,
#     ):
#         self.dataset = dataset
#         self.atoms = atoms
#         self.coordinates = coordinates
#         self.remove_hydrogen = remove_hydrogen
#         self.remove_polar_hydrogen = remove_polar_hydrogen
#         self.set_epoch(None)

#     def set_epoch(self, epoch, **unused):
#         super().set_epoch(epoch)
#         self.epoch = epoch

#     @lru_cache(maxsize=16)
#     def __cached_item__(self, index: int, epoch: int):
#         dd = self.dataset[index].copy()
#         atoms = dd[self.atoms]
#         coordinates = dd[self.coordinates]

#         if self.remove_hydrogen:
#             mask_hydrogen = atoms != "H"
#             atoms = atoms[mask_hydrogen]
#             coordinates = coordinates[mask_hydrogen]
#         if not self.remove_hydrogen and self.remove_polar_hydrogen:
#             end_idx = 0
#             for i, atom in enumerate(atoms[::-1]):
#                 if atom != "H":
#                     break
#                 else:
#                     end_idx = i + 1
#             if end_idx != 0:
#                 atoms = atoms[:-end_idx]
#                 coordinates = coordinates[:-end_idx]
#         dd[self.atoms] = atoms
#         dd[self.coordinates] = coordinates.astype(np.float32)
#         return dd

#     def __getitem__(self, index: int):
#         return self.__cached_item__(index, self.epoch)


import numpy as np
def floyd_warshall(A):
    A = np.array(A, dtype=np.float32)
    n = A.shape[0]
    # dist = np.where(A == 0, np.inf, A)
    dist = np.where(A == 0, 510, A)
    np.fill_diagonal(dist, 0)  

    for k in range(n):
        dist = np.minimum(dist, dist[:, k, np.newaxis] + dist[k, :])
    
    return dist

from rdkit import Chem
def remove_isolated_hydrogens(mol):

    atom_indices_to_remove = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            if len(atom.GetNeighbors()) == 0:
                atom_indices_to_remove.append(atom.GetIdx())
    
    editable_mol = Chem.RWMol(mol)
    
    for idx in reversed(atom_indices_to_remove):
        editable_mol.RemoveAtom(idx)
    
    return editable_mol.GetMol()



class RemoveHydrogenDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        smi,
        remove_hydrogen=False,
        remove_polar_hydrogen=False,
        max_hop=32,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.smi = smi
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.max_hop = max_hop
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        smi = dd[self.smi]
        mol = Chem.MolFromSmiles(smi) 

        if not self.remove_hydrogen:
            mol = Chem.AddHs(mol)

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
            # if "H" in [atom.GetSymbol() for atom in mol.GetAtoms()]:
            #     no_h_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            #     mol = Chem.MolFromSmiles(no_h_smi)
            #     mol = remove_isolated_hydrogens(mol)
            mol_edit = Chem.RWMol(mol)
            for atom in reversed(mol.GetAtoms()):
                if atom.GetSymbol() == 'H':  
                    mol_edit.RemoveAtom(atom.GetIdx())  
            mol = mol_edit
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
        # print("1",[atom.GetSymbol() for atom in mol.GetAtoms()])
        # print("2",atoms)
        assert mol.GetNumAtoms() == len(atoms), ([atom.GetSymbol() for atom in mol.GetAtoms()], atoms,smi)
        #assert [atom.GetSymbol() for atom in mol.GetAtoms()] == atoms, ([atom.GetSymbol() for atom in mol.GetAtoms()], atoms,smi)
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        shortest_path_matrix = floyd_warshall(adjacency_matrix)
        shortest_path_matrix = shortest_path_matrix+1
        shortest_path_matrix[shortest_path_matrix>self.max_hop]=0
        degree = np.sum(adjacency_matrix, axis=1)
        #bond_type = np.zeros_like(adjacency_matrix, dtype=np.int64)
        # for bond in  mol.GetBonds():
        #     start_atom = bond.GetBeginAtomIdx()
        #     end_atom = bond.GetEndAtomIdx() 
            # bond_type_idx = safe_index(possible_bond_type, str(bond.GetBondType()))
            # bond_type[start_atom, end_atom] = bond_type_idx
            # bond_type[end_atom, start_atom] = bond_type_idx

        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd["shortest"] = shortest_path_matrix
        dd["degree"] = degree
        # dd["bond_type"] = bond_type
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class RemoveHydrogenResiduePocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, atoms, residues, coordinates, remove_hydrogen=True):
        self.dataset = dataset
        self.atoms = atoms
        self.residues = residues
        self.coordinates = coordinates
        self.remove_hydrogen = remove_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        residues = dd[self.residues]
        coordinates = dd[self.coordinates]
        if len(atoms) != len(residues):
            min_len = min(len(atoms), len(residues))
            atoms = atoms[:min_len]
            residues = residues[:min_len]
            coordinates = coordinates[:min_len, :]

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            residues = residues[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

        dd[self.atoms] = atoms
        dd[self.residues] = residues
        dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class RemoveHydrogenPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        holo_coordinates,
        remove_hydrogen=True,
        remove_polar_hydrogen=False,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.holo_coordinates = holo_coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        holo_coordinates = dd[self.holo_coordinates]

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
            holo_coordinates = holo_coordinates[mask_hydrogen]
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
                holo_coordinates = holo_coordinates[:-end_idx]
        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    


class RemoveHydrogenLigandDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        holo_coordinates,
        smi,
        remove_hydrogen=True,
        remove_polar_hydrogen=False,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.holo_coordinates = holo_coordinates
        self.smi = smi
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        holo_coordinates = dd[self.holo_coordinates]
        smi = dd[self.smi]
        mol = Chem.MolFromSmiles(smi)

        if not self.remove_hydrogen:
            mol = Chem.AddHs(mol)

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
            holo_coordinates = holo_coordinates[mask_hydrogen]

            mol_edit = Chem.RWMol(mol)
            for atom in reversed(mol.GetAtoms()):
                if atom.GetSymbol() == 'H':  
                    mol_edit.RemoveAtom(atom.GetIdx())  
            mol = mol_edit
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
                holo_coordinates = holo_coordinates[:-end_idx]

        assert mol.GetNumAtoms() == len(atoms), ([atom.GetSymbol() for atom in mol.GetAtoms()], atoms,smi)
        #assert [atom.GetSymbol() for atom in mol.GetAtoms()] == atoms, ([atom.GetSymbol() for atom in mol.GetAtoms()], atoms,smi)
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        shortest_path_matrix = floyd_warshall(adjacency_matrix)
        shortest_path_matrix = shortest_path_matrix+1
        shortest_path_matrix[shortest_path_matrix>32]=0
        degree = np.sum(adjacency_matrix, axis=1)


        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
        dd["shortest"] = shortest_path_matrix.astype(np.int64)
        dd["degree"] = degree
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
