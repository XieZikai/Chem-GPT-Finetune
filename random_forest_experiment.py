from rdkit import Chem
from rdkit.Chem import AllChem
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

fp_gen = AllChem.GetRDKitFPGenerator()


def train_and_evaluate_jsonl(path):
    assert os.path.exists(os.path.join(path, 'train.jsonl'))
    assert os.path.exists(os.path.join(path, 'valid.jsonl'))

    with open(os.path.join(path, 'train.jsonl'), 'r') as train_json:
        json_list = list(train_json)

    train_y = []
    train_fingerprint = []

    for json_str in json_list:
        try:
            pair = json.loads(json_str)
            train_smile = pair['prompt']
            mol = Chem.MolFromSmiles(train_smile)
            fingerprint = fp_gen.GetFingerprint(mol)

            train_y.append(pair['completion'])

            train_fingerprint.append(fingerprint.ToList())
        except:
            pass  # ignore SMILES that cannot be fingerprinted

    valid_y = []
    valid_fingerprint = []

    with open(os.path.join(path, 'valid.jsonl'), 'r') as train_json:
        json_list = list(train_json)

    for json_str in json_list:
        try:
            pair = json.loads(json_str)
            valid_smile = pair['prompt']
            mol = Chem.MolFromSmiles(valid_smile)
            fingerprint = fp_gen.GetFingerprint(mol)
            valid_fingerprint.append(fingerprint.ToList())
            valid_y.append(pair['completion'])
        except:
            pass

    clf = RandomForestClassifier(n_estimators=40)
    clf = clf.fit(train_fingerprint, train_y)

    pred_y = clf.predict(valid_fingerprint)
    accuracy = accuracy_score(valid_y, pred_y)
    f1 = f1_score(valid_y, pred_y, average='weighted')

    return accuracy, f1


def train_and_evaluate_csv(df_train, df_test):
    df = pd.read_csv(df_train)

    train_y = []
    train_fingerprint = []

    for i in range(len(df)):
        pair = df.iloc[i]
        try:
            train_smile = pair['prompt']
            mol = Chem.MolFromSmiles(train_smile)
            fingerprint = fp_gen.GetFingerprint(mol)

            train_y.append(pair['completion'])

            train_fingerprint.append(fingerprint.ToList())
        except:
            pass

    valid_y = []
    valid_fingerprint = []

    df = pd.read_csv(df_test)

    for i in range(len(df)):
        pair = df.iloc[i]
        try:
            valid_smile = pair['prompt']
            mol = Chem.MolFromSmiles(valid_smile)
            fingerprint = fp_gen.GetFingerprint(mol)
            valid_fingerprint.append(fingerprint.ToList())
            valid_y.append(pair['completion'])
        except:
            pass

    clf = RandomForestClassifier(n_estimators=40)
    clf = clf.fit(train_fingerprint, train_y)

    pred_y = clf.predict(valid_fingerprint)
    accuracy = accuracy_score(valid_y, pred_y)
    f1 = f1_score(valid_y, pred_y, average='weighted')

    return accuracy, f1