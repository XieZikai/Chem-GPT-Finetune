import os
import pandas as pd
import numpy as np
import random
from utils import make_output_dir
import pystow


DATA_FILE = 'CSD_EES_DB.csv'


def generate_classification_dataset(column, df=None, num_class=2, split=0.8):
    """
    Column set: [HOMO, LUMO, E(S1), f(S1), E(S2), f(S2), E(S3), f(S3), E(T1), E(T2), E(T3)]
    Generate classification dataset by splitting data in percentile.
    :param
        num_class:
        Class of classification. All dataset will be divided according to the percentiles of class numbers.
    :param
        column:
        Classification target.
    :return:
        train_dataset:
        test_dataset:
    """
    assert column in ['HOMO', 'LUMO', 'E(S1)', 'f(S1)', 'E(S2)', 'f(S2)', 'E(S3)', 'f(S3)', 'E(T1)', 'E(T2)', 'E(T3)']
    if df is None:
        df = pd.read_csv(DATA_FILE)
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    target_array = df[column].to_numpy()
    percentiles = []
    for i in range(num_class):
        percentiles.append(np.percentile(target_array, (i + 1) * 100 / num_class))
    completion = []
    for i in target_array:
        if num_class == 2:
            if i < percentiles[0]:
                completion.append('Low')
            else:
                completion.append('High')
        else:
            for index, percentile in enumerate(percentiles):
                if percentile >= i:
                    completion.append(str(index))
                    break

    df['completion'] = completion
    df['prompt'] = df['SMILES']

    index = list(range(len(df)))
    random.shuffle(index)

    return df[['prompt', 'completion']].iloc[index[:int(split * len(index))], :], \
        df[['prompt', 'completion']].iloc[index[int(split * len(index)):], :],


def generate_classification_dataset_by_equipartition(column='LUMO', df=None, num_class=2, split=0.8):
    """
    Generate classification dataset by splitting data in min-max equipartition.
    :param column:
    :param df:
    :param num_class:
    :param split:
    :return:
    """
    assert column in ['HOMO', 'LUMO', 'E(S1)', 'f(S1)', 'E(S2)', 'f(S2)', 'E(S3)', 'f(S3)', 'E(T1)', 'E(T2)', 'E(T3)']
    if df is None:
        df = pd.read_csv(DATA_FILE)
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    target_array = df[column].to_numpy()
    percentiles = []
    interval = target_array.max() - target_array.min()
    for i in range(num_class):
        percentiles.append(target_array.min() + (i+1) * interval/num_class)
    completion = []
    for i in target_array:
        flag = False
        if num_class == 2:
            if i < percentiles[0]:
                completion.append('Low')
                flag = True
            else:
                completion.append('High')
                flag = True
        else:
            for index, percentile in enumerate(percentiles):
                if percentile + 1e-8 >= i:
                    completion.append(str(index))
                    flag = True
                    break
        if flag is False:
            print(index, percentile)

    df['completion'] = completion
    df['prompt'] = df['SMILES']

    index = list(range(len(df)))
    random.shuffle(index)

    return df[['prompt', 'completion']].iloc[index[:int(split * len(index))], :], \
        df[['prompt', 'completion']].iloc[index[int(split * len(index)):], :],


def generate_core_train_test_by_equipartition(column, core_file_name, num_class=2):
    """
    Generate core-file ablation study dataset by splitting data in min-max equipartition.
    :param column:
    :param core_file_name: Molecules in certain functional group that are taken out as validation set.
    :param num_class:
    :return:
    """
    assert column in ['HOMO', 'LUMO', 'E(S1)', 'f(S1)', 'E(S2)', 'f(S2)', 'E(S3)', 'f(S3)', 'E(T1)', 'E(T2)', 'E(T3)']
    df = pd.read_csv(DATA_FILE)
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    target_array = df[column].to_numpy()
    percentiles = []
    interval = target_array.max() - target_array.min()
    for i in range(num_class):
        percentiles.append(target_array.min() + (i + 1) * interval / num_class)
    completion = []
    for i in target_array:
        flag = False
        if num_class == 2:
            if i < percentiles[0]:
                completion.append('Low')
                flag = True
            else:
                completion.append('High')
                flag = True
        else:
            for index, percentile in enumerate(percentiles):
                if percentile + 1e-8 >= i:
                    completion.append(str(index))
                    flag = True
                    break
        if flag is False:
            print(index, percentile)

    df['completion'] = completion
    df['prompt'] = df['SMILES']
    core = pd.read_csv(core_file_name, header=None)
    core = core[[0]]
    core.columns = ['ID']

    df['completion'] = completion
    df['prompt'] = df['SMILES']
    df = df[['ID', 'completion', 'prompt']]
    test_df = df.merge(core, on='ID', how='right').dropna()
    train_df = pd.concat([df, test_df, test_df]).drop_duplicates(keep=False).dropna()

    return train_df[['prompt', 'completion']], test_df[['prompt', 'completion']]


def generate_core_train_test(column, core_file_name, num_class=2):
    assert column in ['HOMO', 'LUMO', 'E(S1)', 'f(S1)', 'E(S2)', 'f(S2)', 'E(S3)', 'f(S3)', 'E(T1)', 'E(T2)', 'E(T3)']
    df = pd.read_csv(DATA_FILE)
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    target_array = df[column].to_numpy()
    percentiles = []
    for i in range(num_class):
        percentiles.append(np.percentile(target_array, (i + 1) * 100 / num_class))
    completion = []
    for i in target_array:
        if num_class == 2:
            if i < percentiles[0]:
                completion.append('Low')
            else:
                completion.append('High')
        else:
            for index, percentile in enumerate(percentiles):
                if percentile >= i:
                    completion.append(str(index))
                    break

    df['completion'] = completion
    df['prompt'] = df['SMILES']

    core = pd.read_csv(core_file_name, header=None)
    core = core[[0]]
    core.columns = ['ID']

    df['completion'] = completion
    df['prompt'] = df['SMILES']
    df = df[['ID', 'completion', 'prompt']]
    test_df = df.merge(core, on='ID', how='right').dropna()
    train_df = pd.concat([df, test_df, test_df]).drop_duplicates(keep=False).dropna()

    return train_df[['prompt', 'completion']], test_df[['prompt', 'completion']]


def write_file(df_train: pd.DataFrame, df_test: pd.DataFrame, postfix: str = ''):
    """Write a dataframe to a file as json in records form."""
    outdir = make_output_dir(postfix)

    filename_train = os.path.abspath(os.path.join(outdir, "train.jsonl"))
    filename_valid = os.path.abspath(os.path.join(outdir, "valid.jsonl"))

    df_train.to_json(filename_train, orient="records", lines=True, force_ascii=False)
    df_test.to_json(filename_valid, orient="records", lines=True, force_ascii=False)


def json_to_csv(data_path):
    def replace_binary_name(a):
        if a == 'High' or a == 'high':
            return 1
        elif a == 'Low' or a == 'low':
            return 0
        else:
            return a
    with open(os.path.join(data_path, 'valid.jsonl'), 'r') as json_file:
        json_list = list(json_file)
        json_list = [eval(i) for i in json_list]
        for i, json in enumerate(json_list):
            json_list[i]['completion'] = replace_binary_name(json_list[i]['completion'])
        valid_df = pd.DataFrame(json_list)
        valid_df.columns = ['smiles', 'target']
        valid_df.to_csv(os.path.join(data_path, 'valid.csv'), index=False)

    with open(os.path.join(data_path, 'train.jsonl'), 'r') as json_file:
        json_list = list(json_file)
        json_list = [eval(i) for i in json_list]
        for i, json in enumerate(json_list):
            json_list[i]['completion'] = replace_binary_name(json_list[i]['completion'])
        train_df = pd.DataFrame(json_list)
        train_df.columns = ['smiles', 'target']
        train_df.to_csv(os.path.join(data_path, 'train.csv'), index=False)


def get_photoswitch_data() -> pd.DataFrame:
    """Return the photoswitch data as a pandas DataFrame.
    References:
        [GriffithsPhotoSwitches] `Griffiths, K.; Halcovitch, N. R.; Griffin, J. M. Efficient Solid-State Photoswitching of Methoxyazobenzene in a Metal–Organic Framework for Thermal Energy Storage. Chemical Science 2022, 13 (10), 3014–3019. <https://doi.org/10.1039/d2sc00632d>`_
    """
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "photoswitches",
            url="https://www.dropbox.com/s/z5z9z944cc060x9/photoswitches.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        )
        .drop_duplicates(subset=["SMILES"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    BASE_DIR = r'C:\Users\darkn\PycharmProjects\ChemGPT\out\new_data_gpt\ablation\new_core_ablation'
    folders = os.listdir(BASE_DIR)
    for folder in folders:
        if not os.path.exists(os.path.join(BASE_DIR, folder)):
            continue
        sub_folder = os.path.join(BASE_DIR, folder)
        if 'valid.jsonl' in os.listdir(sub_folder):
            json_to_csv(sub_folder)
