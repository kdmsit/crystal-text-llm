import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from io import StringIO

# Step 1: Read the input CSV
input_csv = 'data/mpts_52/test.csv' # Replace with your actual filename

df = pd.read_csv(input_csv)

# Step 2: Function to extract pretty_formula from CIF string
def get_pretty_formula(cif_string):
    try:
        parser = CifParser(StringIO(cif_string))
        structure = parser.get_structures(primitive=False)[0]
        return structure.composition.reduced_formula
    except Exception as e:
        print(f"Error parsing CIF: {e}")
        return None

# Step 3: Generate pretty_formula for each row
df['pretty_formula'] = df['cif'].apply(get_pretty_formula)

# Step 4: Save to a new CSV
output_csv = 'data/mpts_52/tagged/test.csv'
df.to_csv(output_csv, index=False)

print(f"Saved updated CSV to {output_csv}")
