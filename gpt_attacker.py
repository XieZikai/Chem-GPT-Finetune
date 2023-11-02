import numpy as np
import numbers

import openai
import tiktoken
import pandas as pd
from rdkit import Chem, DataStructs
from copy import copy
import re
import os
import json
from utils import replace_smiles_with_missing, match_smart, match_smart_by_group, SMART_LIST
from tqdm import tqdm


class Attacker:
    """
    GPT-model attacker to conduct atom-replacement attack for ablation study.
    Call method replacement_test to conduct experiment.
    """
    def __init__(self, model_id, model_name='ada'):
        """

        :param model_id: Pretrained GPT model id to attack.
        :param model_name: Pretrained GPT model name.
        """
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.model_id = model_id
        self.token_list = self.get_all_tokens()
        self.word_list = [self.decode_int(token) for token in self.token_list]

    def get_all_tokens(self, source_file='CSD_EES_DB.csv'):
        df = pd.read_csv(source_file)
        token_list = []
        for smile in df['SMILES']:
            encodings = self.encoding.encode(str(smile))
            for encoding in encodings:
                # if encoding not in token_list and (reject not in encoding for reject in reject_list):
                if encoding not in token_list:
                    token_list.append(encoding)
        return token_list

    def decode_int(self, integer: int):
        return self.encoding.decode_single_token_bytes(integer).decode('utf-8')

    def tokenize(self, sequence) -> (list, list):
        """
        Tokenize the sequence into a list of tokens using tiktoken.
        :param sequence:
        :return:
        """
        if isinstance(sequence, list):
            sentence = ''.join(sequence)
        else:
            sentence = sequence
        encodings = self.encoding.encode(sentence)
        words = [self.decode_int(token) for token in encodings]
        return encodings, words

    def word_distance(self, token1, token2):
        """
        Alphabet distance of two words with same length.
        :param token1:
        :param token2:
        :return:
        """
        if isinstance(token1, int):
            token1 = self.decode_int(token1)
        if isinstance(token1, int):
            token2 = self.decode_int(token2)
        assert isinstance(token1, str) and isinstance(token2, str) and len(token1) == len(token2)
        result = 0
        for i in range(len(token1)):
            result += abs(ord(token1[i]) - ord(token2[i]))
        return result

    def synonyms(self, x, pos_x) -> list:
        """
        Return a list of synonym words for target token.
        :param x: original sample for context.
        :param pos_x: place in x for synonym replacement. x and i are used to check SMILES availability.
        :return:
        """
        token = x[pos_x]
        if isinstance(token, int):
            token = self.decode_int(token)

        result_list_word = [i for i in self.word_list if len(i) == len(token)]
        result_list = []

        # Availability check
        for word in result_list_word:
            _, x_new = self.tokenize(x)
            x_new[pos_x] = word
            if self.check_smiles_valid(self.word_list_to_smiles(x_new)):
                result_list.append(word)

        result_list_score = [self.word_distance(i, token) for i in result_list]
        result_list = list(zip(result_list_score, result_list))
        result_list.sort()
        if result_list[0][0] == 0:
            result_list = result_list[1:]
        return result_list

    @staticmethod
    def check_smiles_valid(smiles):
        m = Chem.MolFromSmiles(smiles)
        return False if m is None else True

    @staticmethod
    def word_list_to_smiles(word_list):
        return ''.join(word_list)

    def token_list_to_smiles(self, token_list):
        word_list = [self.decode_int(token) for token in token_list]
        return self.word_list_to_smiles(word_list)

    def replace_test_atom_level(self, x, real_class, replace_num=0.2):

        atom_set = ['B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I']
        x_replaces = []
        test_results = []

        for atom in atom_set:
            x_list = copy(x).split(atom)
            if len(x_list) == 1:
                continue

            for i in range(len(x_list) - 1):
                x_replace = copy(x_list)
                # print('x: ',x_replace)
                x_replace[i] = x_replace[i] + '<missing>'
                x_replace = atom.join(x_replace)
                missing_place = re.search('<missing>', x_replace)
                x_replace = x_replace[: missing_place.span()[1]] + x_replace[missing_place.span()[1] + len(atom):]
                x_replaces.append(x_replace)
                test_results.append(self.attack_criteria(x_replace, real_class))

        return x_replaces, test_results

    def replace_test_token_level(self, x, real_class, random_rate=0.1):
        tokens, x_star = self.tokenize(x)
        word_count = len(tokens)
        indices = np.random.choice(range(word_count), size=int(word_count * random_rate), replace=False)

        x_replaces = []
        test_results = []

        for i in indices:
            # print(x_star[i])
            if x_star[i].isalpha():
                x_i = copy(x_star)
                x_i[i] = ''.join(['<missing>' for _ in range(len(x_star[i]))])
                x_replaces.append(''.join(x_i))
                test_results.append(self.attack_criteria(x_i, real_class))

        return x_replaces, test_results

    def attack(self, x, real_class, random_rate=0.3):
        """
        Conduct attack on the test_sample.

        :param x:
        :param random_rate:
        :param real_class:
        :return:
        """
        scores = []
        tokens, x_star = self.tokenize(x)
        word_count = len(tokens)
        indices = np.random.choice(range(word_count), size=int(word_count * random_rate), replace=False)
        for i in indices:
            synonyms = self.synonyms(x, i)
            w = np.random.choice(range(len(synonyms)))
            x_star[i] = synonyms[w][1]
            if self.attack_criteria(x_star, real_class):
                break

        for i in indices:
            x_i = copy(x_star)
            x_i[i] = x[i]
            score_i = self.score(x_i, x)
            if self.attack_criteria(x_i, real_class):
                scores.append((score_i, x[i], i))

        scores = sorted(scores, reverse=True)

        for score_x_pair in scores:
            score_i, x_word_i, i = score_x_pair
            x_t = copy(x_star)
            x_t[i] = x_word_i
            if not self.attack_criteria(x_t, real_class):
                break
            x_star = x_t

        return x_star

    @staticmethod
    def score(x_i, x) -> numbers.Number:
        """
        Scoring the adversarial sample and the original sample based on fingerprint similarity.

        :param x_i: SMILES string
        :param x: SMILES string
        :return:
        """
        fingerprint_i = Chem.RDKFingerprint(Chem.MolFromSmiles(''.join(x_i)))
        fingerprint = Chem.RDKFingerprint(Chem.MolFromSmiles(''.join(x)))
        return DataStructs.FingerprintSimilarity(fingerprint_i, fingerprint)

    def attack_criteria(self, attack_sample, real_class) -> bool:
        """
        Check if model(real_sample) != model(attack_sample).
        Need to call OpenAI model once.

        :param attack_sample:
        :param real_class:
        :return:
        """
        if isinstance(attack_sample, list):
            prompt = ''.join(attack_sample)
        else:
            prompt = attack_sample
        evaluation = openai.Completion.create(model=self.model_id, prompt=prompt, max_tokens=1, temperature=0)
        return not (real_class == evaluation['choices'][0]['text'])

    def evaluate(self, prompt):
        evaluation = openai.Completion.create(model=self.model_id, prompt=prompt, max_tokens=1, temperature=0)
        return evaluation['choices'][0]['text']

    def genetic_optimize(self):
        pass

    #
    def replacement_test(self, path, start_from=0):
        """
        Ablation study: replacement test.

        When openai API timeout, the script will stop and record all previous result. Please use 'start_from' to
        resume the test.

        :param path: data path
        :param start_from: starting point for resuming the test
        :return:
        """
        with open(os.path.join(path, 'valid.jsonl'), 'r') as json_file:
            json_list = list(json_file)

        instance_list = []
        attacked_list = []
        count = start_from

        for json_str in tqdm(json_list[start_from:]):
            try:

                pair = json.loads(json_str)
                y_pred = self.evaluate(pair['prompt'])
                if y_pred != pair['completion']:
                    continue

                instance_list.append(pair)

                matches = match_smart(pair['prompt'])
                for i, match in enumerate(matches):
                    if len(match) == 0:
                        continue
                    for single_match in match:
                        print(single_match)
                        attacked_smiles = replace_smiles_with_missing(pair['prompt'], single_match)
                        if attacked_smiles is not None:
                            attacked_list.append({'prompt': pair['prompt'],
                                                  'replaced': attacked_smiles,
                                                  'completion_true': pair['completion'],
                                                  'completion_pred': self.evaluate(attacked_smiles),
                                                  'replace_index': i})
                        else:
                            print('Molecule {} cannot be tested!'.format(pair['prompt']))
                            break
                count += 1
            except Exception as e:
                print(e)
                print(count)
                pd.DataFrame(attacked_list).to_csv('replacement_result_until_{}.csv'.format(count))

                return attacked_list

        print(count)
        pd.DataFrame(attacked_list).to_csv('lumo_replacement_result_until_{}.csv'.format(count))
        return attacked_list


if __name__ == '__main__':
    path = r'C:\Users\shibi\PycharmProjects\gptchem\out\new_data_gpt\small_molecule\20230701_152139__0.4_LUMO_1'
    model_id = 'ada:ft-birmingham-digital-chemistry-2023-07-03-13-39-02'
    attacker = Attacker(model_id)

    result = attacker.replacement_test(path)
