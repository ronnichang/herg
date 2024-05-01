from rdkit import Chem
from rdkit.Chem import AllChem


# Function to convert SMILES to Morgan fingerprints
def smiles_to_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

