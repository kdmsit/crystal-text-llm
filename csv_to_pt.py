import torch
import argparse
import pandas as pd
from tqdm import tqdm
from data_utils import process_one
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def task(cif_str):
    return process_one(cif_str, True, False,'crystalnn', False, 0.01)


def main(args):
    data_path = args.data_path
    file_name = data_path.replace('.csv', '')
    print(file_name)

    n_atom, x_coord, a_type, length, angle = [], [], [], [], []
    df_data = pd.read_csv(data_path)
    executor = ThreadPoolExecutor(max_workers=16)

    pbar = tqdm(total=len(df_data), desc="Generating Samples")
    for index, row in df_data.iterrows():
        # print(index)
        # if index in [152,154,291, 294,295]:
        #     continue
        cif_str = str(row['cif'])
        future = executor.submit(task, cif_str)
        try:
            frac_coords, atom_types, lengths, angles, num_atoms = future.result(timeout=10)
            # frac_coords, atom_types, lengths, angles, num_atoms = process_one(cif_str, True, False,'crystalnn', False, 0.01)
        except Exception as e:
            print(e)
            continue
        except TimeoutError:
            # print(f"Task {i} took too long. Skipping...")
            skipped_count = skipped_count + 1
            if skipped_count % 10 == 0:
                print(f"Total {skipped_count} are skipped so far...")

        num_atoms = torch.tensor([num_atoms])
        frac_coords = torch.tensor(frac_coords)
        atom_types = torch.tensor(atom_types)
        lengths = torch.tensor(lengths)
        angles = torch.tensor(angles)

        n_atom.append(num_atoms)
        x_coord.append(frac_coords)
        a_type.append(atom_types)
        length.append(lengths.view(1, 3))
        angle.append(angles.view(1, 3))
        pbar.update(1)
    n_atom = torch.cat(n_atom, dim=0)
    x_coord = torch.cat(x_coord, dim=0)
    a_type = torch.cat(a_type, dim=0)
    length = torch.cat(length, dim=0)
    angle = torch.cat(angle, dim=0)

    n_atom = n_atom.unsqueeze(0)
    x_coord = x_coord.unsqueeze(0)
    a_type = a_type.unsqueeze(0)
    length = length.unsqueeze(0)
    angle = angle.unsqueeze(0)

    print(n_atom.size())
    print(x_coord.size())
    print(a_type.size())
    print(length.size())
    print(angle.size())

    path = os.path.join("saved_gen_"+str(file_name)+".pt")
    torch.save({
        "frac_coords": x_coord,
        "num_atoms": n_atom,
        "atom_types": a_type,
        "lengths": length,
        "angles": angle,
    }, path)
    print("Saved to file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args)