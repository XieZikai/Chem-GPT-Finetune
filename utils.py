from rdkit import Chem
import copy
import re
import pandas as pd
import os
import time


BASE_OUTDIR = './data/out'

SMART_LIST = [
    '[NX1]#[CX2]',
    '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]',
    '[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]',
    '[NX3][$(C=C),$(cc)]',
    '[#6][CX3](=O)[#6]',
    '[OX1]=CN',
    '[CX3](=[OX1])O',
    '*-[N;D2]=[N;D2]-[C;D1;H3]',
    '*-[N;D2]=[N;D1]',
    '*-[N;D2]#[N;D1]',
    '*-[C;D2]#[N;D1]',
    '*-[S;D1]',
    '*=[S;D1]',
    '[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]',
    '*-[S;D4](=O)(=O)-[O;D1]',
    '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]',
    '[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]',
    '[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]',
    '[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]',
    '*-[C;D2]#[C;D1;H]',
    '*-[#9,#17,#35,#53]'
]


def make_output_dir(run_name):
    outdir = os.path.abspath(os.path.join(BASE_OUTDIR, time.strftime("%Y%m%d_%H%M%S")))
    if run_name is not None:
        outdir = f"{outdir}_{run_name}"
    os.makedirs(outdir, exist_ok=True)
    return outdir


def startwith(string, atom):
    if atom == 'C':
        return string.startswith('C') or string.startswith('c')
    return string.startswith(atom)


def replace_smiles_with_missing(smiles, atom_indices):
    keys = ['=', '#']
    mol = Chem.MolFromSmiles(smiles)

    atom_list = mol.GetAtoms()
    atom_string_list = [atom.GetSymbol() for atom in mol.GetAtoms()]

    new_smile_list = []
    pointer = 0
    atom_pointer = 0
    atom_indices = sorted(atom_indices)

    try:
        for atom_index in atom_indices:
            while atom_pointer < atom_index:
                sub_smiles = smiles[pointer:]

                if startwith(sub_smiles, atom_string_list[atom_pointer]):
                    if atom_list[atom_pointer].IsInRing():
                        new_smile_list.append(atom_string_list[atom_pointer].lower())
                    else:
                        new_smile_list.append(atom_string_list[atom_pointer])
                    pointer += len(atom_string_list[atom_pointer])
                    atom_pointer += 1
                else:
                    new_smile_list.append(smiles[pointer])
                    pointer += 1

            if new_smile_list[-1] in keys:
                new_smile_list = new_smile_list[:-1]

            sub_smiles = smiles[pointer:]

            while not startwith(sub_smiles, atom_string_list[atom_index]):
                if smiles[pointer] not in keys:
                    new_smile_list.append(smiles[pointer])
                    pointer += 1
                else:
                    pointer += 1
                sub_smiles = smiles[pointer:]

            new_smile_list.append('<missing>')
            pointer += len(atom_string_list[atom_index])
            atom_pointer += 1
            if smiles[pointer] in keys:
                pointer += 1

        new_smile_list.append(smiles[pointer:])
    except IndexError:
        return None

    return ''.join(new_smile_list)


def replace_one_atom(smiles, remove=False):
    atom_set = ['B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I']
    removed_list = []
    for atom in atom_set:
        x_list = copy.copy(smiles).split(atom)
        if len(x_list) == 1:
            continue

        for i in range(len(x_list) - 1):
            x_replace = copy.copy(x_list)
            # print('x: ',x_replace)
            x_replace[i] = x_replace[i] + '<missing>'
            x_replace = atom.join(x_replace)
            missing_place = re.search('<missing>', x_replace)
            x_replace = x_replace[: missing_place.span()[1]] + x_replace[missing_place.span()[1] + len(atom):]
            if remove:
                x_replace = x_replace.replace('<missing>', '')
            removed_list.append(x_replace)
    return removed_list


def match_smart(smiles, smart_list=None):
    if smart_list is None:
        smart_list = SMART_LIST
    matches = []
    for smart in smart_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            match = Chem.MolFromSmiles(smiles).GetSubstructMatches(Chem.MolFromSmarts(smart))
            matches.append(match)
    return matches


def match_smart_by_group(smiles, smart):
    matches = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            match = Chem.MolFromSmiles(smile).GetSubstructMatches(Chem.MolFromSmarts(smart))
            matches.append(match)
    return matches


def write_file(df_train: pd.DataFrame, df_test: pd.DataFrame, postfix: str = ''):
    """Write a dataframe to a file as json in records form."""
    outdir = make_output_dir(postfix)

    filename_train = os.path.abspath(os.path.join(outdir, "train.jsonl"))
    filename_valid = os.path.abspath(os.path.join(outdir, "valid.jsonl"))

    df_train.to_json(filename_train, orient="records", lines=True, force_ascii=False)
    df_test.to_json(filename_valid, orient="records", lines=True, force_ascii=False)