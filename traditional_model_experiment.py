from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors
import os
import json
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

fp_gen = AllChem.GetRDKitFPGenerator()


def train_and_evaluate_jsonl(path):
    """
    Train and evaluate random forest classifier for dataset in certain path.
    :param path: Dataset path, train.jsonl and valid.jsonl must be included.
    :return:
    """
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
    """
    Train and evaluate random forest classifier for dataset in the form of pandas DataFrame.
    :param df_train: Training set dataframe.
    :param df_test: Validation set dataframe.
    :return:
    """

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


def train_and_evaluate_jsonl_traditional(path):
    """
    Train and evaluate random forest classifier for dataset in certain path.
    :param path: Dataset path, train.jsonl and valid.jsonl must be included.
    :return:
    """
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
            fingerprint = list(CalcMolDescriptors(mol).values())
            train_y.append(pair['completion'])

            train_fingerprint.append(fingerprint)
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
            fingerprint = list(CalcMolDescriptors(mol).values())
            valid_fingerprint.append(fingerprint)
            valid_y.append(pair['completion'])
        except:
            pass

    train_fingerprint = np.array(train_fingerprint)
    valid_fingerprint = np.array(valid_fingerprint)

    k_best_selector = SelectKBest(score_func=f_classif, k=20)
    train_selected = k_best_selector.fit_transform(train_fingerprint, train_y)
    valid_selected = k_best_selector.fit_transform(valid_fingerprint, valid_y)

    clf = RandomForestClassifier(n_estimators=40)
    clf = clf.fit(train_selected, train_y)

    pred_y = clf.predict(valid_selected)
    accuracy = accuracy_score(valid_y, pred_y)
    f1 = f1_score(valid_y, pred_y, average='weighted')

    return accuracy, f1


def train_and_evaluate_jsonl_traditional_svm(path, min_columns=20, k=2):
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn.preprocessing import MinMaxScaler

    assert os.path.exists(os.path.join(path, 'train.jsonl'))
    assert os.path.exists(os.path.join(path, 'valid.jsonl'))

    with open(os.path.join(path, 'train.jsonl'), 'r') as train_json:
        json_list = list(train_json)

    train_y = []
    train_descriptor = []

    for json_str in json_list:
        try:
            pair = json.loads(json_str)
            train_smile = pair['prompt']
            mol = Chem.MolFromSmiles(train_smile)
            descriptor = CalcMolDescriptors(mol)

            dropped_columns = []
            for i in descriptor:
                if descriptor[i] is None:
                    dropped_columns.append(i)
            if min_columns > len(descriptor) - len(dropped_columns):
                continue
            train_descriptor.append(descriptor)
            train_y.append(pair['completion'])
        except:
            pass  # ignore SMILES that cannot be fingerprinted

    train_descriptor = pd.DataFrame(train_descriptor)

    valid_y = []
    valid_descriptor = []

    with open(os.path.join(path, 'valid.jsonl'), 'r') as train_json:
        json_list = list(train_json)

    for json_str in json_list:
        try:
            pair = json.loads(json_str)
            valid_smile = pair['prompt']
            mol = Chem.MolFromSmiles(valid_smile)
            descriptor = CalcMolDescriptors(mol)

            dropped_columns = []
            for i in descriptor:
                if descriptor[i] is None:
                    dropped_columns.append(i)
            if min_columns > len(descriptor) - len(dropped_columns):
                continue
            valid_descriptor.append(descriptor)
            valid_y.append(pair['completion'])
        except:
            pass

    valid_descriptor = pd.DataFrame(valid_descriptor)

    indices_to_drop = []
    for index, i in enumerate(train_descriptor.isna().any()):
        if i is True:
            indices_to_drop.append(index)
    for index, i in enumerate(valid_descriptor.isna().any()):
        if i is True:
            indices_to_drop.append(index)
    indices_to_drop = list(set(indices_to_drop))

    train_descriptor = train_descriptor.drop(train_descriptor.columns[indices_to_drop], axis=1)
    train_descriptor = train_descriptor.to_numpy()

    selector = SelectPercentile(score_func=f_classif, percentile=k)
    X_train_selected = selector.fit_transform(train_descriptor, train_y)

    scaler = MinMaxScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)

    valid_descriptor = valid_descriptor.drop(valid_descriptor.columns[indices_to_drop], axis=1)
    valid_descriptor = valid_descriptor.to_numpy()

    X_valid_selected = selector.transform(valid_descriptor)
    X_valid_selected = scaler.transform(X_valid_selected)

    from sklearn.svm import SVC

    clf = SVC()
    clf = clf.fit(X_train_selected, train_y)

    pred_y = clf.predict(X_valid_selected)
    accuracy = accuracy_score(valid_y, pred_y)
    f1 = f1_score(valid_y, pred_y, average='weighted')
    return accuracy, f1
