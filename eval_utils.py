import itertools
import numpy as np
import torch
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from pathlib import Path

import smact
from smact.screening import pauling_test

from constants import CompScalerMeans, CompScalerStds
from data_utils import StandardScaler
# from pl_data.dataset import TensorCrystDataset
# from pl_data.datamodule import worker_init_fn

from torch_geometric.data import DataLoader

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)


chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


def load_data(file_path):
    # if file_path[-3:] == 'npy':
    #     data = np.load(file_path, allow_pickle=True).item()
    #     for k, v in data.items():
    #         if k == 'input_data_batch':
    #             for k1, v1 in data[k].items():
    #                 data[k][k1] = torch.from_numpy(v1)
    #         else:
    #             data[k] = torch.from_numpy(v).unsqueeze(0)
    # else:
    data = torch.load(file_path)
    return data


def get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    # print('frac coords',frac_coords.size())
    # print('atom_types',atom_types.size())
    # print('num_atoms',num_atoms.size())
    # print('lengths',lengths.size())
    # print('angles',angles.size())
    #
    # print(frac_coords.size(0))
    # print(atom_types.size(0))
    # print(num_atoms.sum())

    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        # print(cur_atom_types)
        # cur_atom_types = torch.where(
        #     (cur_atom_types < 0) | (cur_atom_types > 118),
        #     torch.tensor(1, dtype=cur_atom_types.dtype),
        #     cur_atom_types
        # )
        # all_valid = ((cur_atom_types >= 0) & (cur_atom_types <= 119)).all().item()
        # print(all_valid)
        # if not all_valid:
        #     continue
        # print(cur_atom_types)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.detach().cpu().numpy(),
            'lengths': cur_lengths.detach().cpu().numpy(),
            'angles': cur_angles.detach().cpu().numpy(),
        })

        start_idx = start_idx + num_atom
    print(len(crystal_array_list))
    return crystal_array_list


def smact_validity(comp, count,use_pauling_test=True,include_alloys=True):
    s=[]

    for elem in comp:
        try:
            if elem<=0 or elem>119:
                elem=1
            chem_symbol = chemical_symbols[elem]
            s.append(chem_symbol)
        except Exception:
            continue
    elem_symbols = tuple(s)


    # elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    # print(elem_symbols)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


# def prop_model_eval(eval_model_name, crystal_array_list):
#
#     model_path = get_model_path(eval_model_name)
#
#     model, _, _ = load_model(model_path)
#     cfg = load_config(model_path)
#
#     dataset = TensorCrystDataset(
#         crystal_array_list, cfg.data.niggli, cfg.data.primitive,
#         cfg.data.graph_method, cfg.data.preprocess_workers,
#         cfg.data.lattice_scale_method)
#
#     dataset.scaler = model.scaler.copy()
#
#     loader = DataLoader(
#         dataset,
#         shuffle=False,
#         batch_size=256,
#         num_workers=0,
#         worker_init_fn=worker_init_fn)
#
#     model.eval()
#
#     all_preds = []
#
#     for batch in loader:
#         preds = model(batch,teacher_forcing=None, training=None)
#         model.scaler.match_device(preds)
#         scaled_preds = model.scaler.inverse_transform(preds)
#         all_preds.append(scaled_preds.detach().cpu().numpy())
#
#     all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
#     return all_preds.tolist()


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        # 'amsd_recall': np.mean(struc_recall_dist),
        # 'amsd_precision': np.mean(struc_precision_dist),
        # 'amcd_recall': np.mean(comp_recall_dist),
        # 'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict
