"""chem.py - RDKit featurisation: standardisation, InChIKey blocks (join key), Morgan fingerprints, descriptors,
Bemis-Murcko scaffolds (leak-free group splitting)."""
import numpy as np
import pandas as pd

from utils import LOG, tq, check_memory

_RD = {}


def _rd():
    if not _RD:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp.*")
        _RD.update(Chem=Chem, Descriptors=Descriptors, gen=None, std=rdMolStandardize, scaf=MurckoScaffold,
                   fpgen=rdFingerprintGenerator)
    return _RD


def standardize(smiles):
    """Largest fragment + neutralisation. Returns RDKit Mol or None."""
    rd = _rd()
    try:
        m = rd["Chem"].MolFromSmiles(str(smiles))
        if m is None:
            return None
        m = rd["std"].LargestFragmentChooser().choose(m)
        m = rd["std"].Uncharger().uncharge(m)
        return m
    except Exception:
        return None


def ik14_from_mol(m):
    try:
        from rdkit.Chem import inchi
        k = inchi.MolToInchiKey(m)
        return k.split("-")[0] if k else None
    except Exception:
        return None


def smiles_table(smiles_list, desc="standardise SMILES"):
    """-> DataFrame(smiles, std_smiles, ik14, scaffold). ik14 (InChIKey connectivity block) is the cross-dataset join key."""
    rd = _rd()
    rows = []
    for s in tq(list(smiles_list), desc):
        m = standardize(s)
        if m is None:
            rows.append((s, None, None, None))
            continue
        try:
            sc = rd["scaf"].MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        rows.append((s, rd["Chem"].MolToSmiles(m), ik14_from_mol(m), sc or "NOSCAF"))
    return pd.DataFrame(rows, columns=["smiles", "std_smiles", "ik14", "scaffold"])


def ik14_from_inchikey(series):
    return series.astype(str).str.split("-").str[0].where(series.notna())


def morgan_fp(std_smiles, nbits=2048, radius=2, count=True, desc="Morgan fingerprints"):
    rd = _rd()
    gen = rd["fpgen"].GetMorganGenerator(radius=radius, fpSize=nbits)
    X = np.zeros((len(std_smiles), nbits), np.uint8 if not count else np.uint8)
    for i, s in enumerate(tq(list(std_smiles), desc)):
        m = rd["Chem"].MolFromSmiles(s) if s else None
        if m is None:
            continue
        v = gen.GetCountFingerprintAsNumPy(m) if count else gen.GetFingerprintAsNumPy(m)
        X[i] = np.minimum(v, 255)
    return X


def descriptors(std_smiles, desc="RDKit descriptors"):
    rd = _rd()
    names = [n for n, _ in rd["Descriptors"].descList]
    fns = [f for _, f in rd["Descriptors"].descList]
    X = np.full((len(std_smiles), len(names)), np.nan, np.float32)
    for i, s in enumerate(tq(list(std_smiles), desc)):
        m = rd["Chem"].MolFromSmiles(s) if s else None
        if m is None:
            continue
        for j, f in enumerate(fns):
            try:
                v = f(m)
                X[i, j] = v if np.isfinite(v) and abs(v) < 1e30 else np.nan
            except Exception:
                pass
    X[~np.isfinite(X)] = np.nan
    X[np.abs(X) > 1e8] = np.nan          # Ipc / BertzCT-style overflow values
    return X, names


def substructure_for_bit(std_smiles, bit, nbits=2048, radius=2):
    """Explainability: SMILES fragment that sets a given Morgan bit in this molecule."""
    rd = _rd()
    from rdkit.Chem import rdFingerprintGenerator, rdmolops
    m = rd["Chem"].MolFromSmiles(std_smiles)
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(m, additionalOutput=ao)
    info = ao.GetBitInfoMap()
    if bit not in info:
        return None
    atom, rad = info[bit][0]
    if rad == 0:
        return rd["Chem"].MolFragmentToSmiles(m, atomsToUse=[atom], canonical=True)
    env = rdmolops.FindAtomEnvironmentOfRadiusN(m, rad, atom)
    amap = {}
    sub = rdmolops.PathToSubmol(m, env, atomMap=amap)
    return rd["Chem"].MolToSmiles(sub)


def maccs(std_smiles, desc="MACCS keys"):
    from rdkit.Chem import MACCSkeys
    rd = _rd()
    X = np.zeros((len(std_smiles), 167), np.float32)
    for i, s in enumerate(tq(list(std_smiles), desc)):
        m = rd["Chem"].MolFromSmiles(s) if s else None
        if m is None:
            continue
        X[i] = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
    return X


def _count_fp(std_smiles, gen, nbits, desc):
    rd = _rd()
    X = np.zeros((len(std_smiles), nbits), np.float32)
    for i, s in enumerate(tq(list(std_smiles), desc)):
        m = rd["Chem"].MolFromSmiles(s) if s else None
        if m is not None:
            X[i] = np.minimum(gen.GetCountFingerprintAsNumPy(m), 255)
    return X


def atompair_counts(std_smiles, nbits=1024):
    rd = _rd()
    return _count_fp(std_smiles, rd["fpgen"].GetAtomPairGenerator(fpSize=nbits), nbits, "atom-pair counts")


def torsion_counts(std_smiles, nbits=512):
    rd = _rd()
    return _count_fp(std_smiles, rd["fpgen"].GetTopologicalTorsionGenerator(fpSize=nbits), nbits, "torsion counts")
