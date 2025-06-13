from collections import Counter
import argparse
import os
import json

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core import Lattice, Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

from eval_utils import (smact_validity, structure_validity, CompScaler, get_fp_pdist, load_data, get_crystals_list, compute_cov)


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp_20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'mpts_52': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon_24': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perov_5': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp_20': {'struc': 0.4, 'comp': 10.},
    'mpts_52': {'struc': 0.4, 'comp': 10.},
    'carbon_24': {'struc': 0.2, 'comp': 4.},
    'perov_5': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):
    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        # print(self.frac_coords)
        # print(self.atom_types)
        # print(self.lengths)
        # print(self.angles)
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                length = self.lengths.tolist()
                angle = self.angles.tolist()
                # self.atom_types = [item[0] for item in self.atom_types]
                frac_cord = self.frac_coords
                self.structure = Structure(lattice=Lattice.from_parameters(*(length + angle)), species=self.atom_types,coords=frac_cord, coords_are_cartesian=False)
                self.constructed = True
                # print("Constructed",self.constructed)
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
                # if self.structure.volume < 0.1:
                #     self.constructed = False
                #     self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)



class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        self.valid_samples = valid_crys
        # if len(valid_crys) >= n_samples:
        #     sampled_indices = np.random.choice(
        #         len(valid_crys), n_samples, replace=False)
        #     self.valid_samples = [valid_crys[i] for i in sampled_indices]
        # else:
        #     raise Exception(
        #         f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        # metrics.update(self.get_comp_diversity())
        # metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics


def get_file_paths(root_path, task, label='', suffix='pt'):
    if label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path):
    data = load_data(file_path)
    print(data['atom_types'][0].size())
    crys_array_list = get_crystals_list(data['frac_coords'][0],
                                        data['atom_types'][0],
                                        data['lengths'][0],
                                        data['angles'][0],
                                        data['num_atoms'][0])

    return crys_array_list

def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict)


def task(x):
    return Crystal(x)


def main(args):
    all_metrics = {}
    # dataset = args.root_path.split('/')[1]
    # print(dataset)
    eval_model_name = 'mp_20'

    out = open("result.txt", "a")

    if 'gen' in args.tasks:
        # gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        gen_file_path = args.root_path
        print(gen_file_path)
        crys_array_list = get_crystal_array_list(gen_file_path)

        # gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        gen_crys =[]
        for i in range(len(crys_array_list)):
            print(i)
            if i==1405:
                continue
            gen_crys.append(Crystal(crys_array_list[i]))

        # gen_path = str(args.root_path)+'generated/'
        # if not os.path.exists(gen_path):
        #     os.makedirs(gen_path)

        # print("Save the generated Images...")
        # for i in range(len(gen_crys)):
        #     crystal = gen_crys[i]
        #     crystal.structure.to(filename = gen_path + str(i)+".cif")
        # print("Saved the generated Images...[DONE]")

        csv = pd.read_csv(args.gt_file)
        gt_crys = p_map(get_gt_crys_ori, csv['cif'])

        gen_evaluator = GenEval(gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    for k, v in all_metrics.items():
        print(k, round(v,4))
        line = str(k)+' '+ str(round(v,4))
        out.writelines(line)
        out.writelines("\n")
    out.writelines("\n")
    out.writelines("\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--multi_eval', action='store_true')
    parser.add_argument('--gt_file', default='')
    args = parser.parse_args()
    main(args)
    # main('gen/','','recon')
