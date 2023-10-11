import chemprop
import os

BASE_DIR = r'C:\Users\darkn\PycharmProjects\ChemGPT\out\new_data_gpt\ablation\new_core_ablation'

if __name__ == "__main__":

    for folder in os.listdir(BASE_DIR):
        if not os.path.exists(os.path.join(BASE_DIR, folder)):
            continue
        arguments = [
            '--data_path', os.path.join(BASE_DIR, folder + '/train.csv'),
            '--separate_test_path', os.path.join(BASE_DIR, folder + '/train.csv'),
            '--separate_val_path', os.path.join(BASE_DIR, folder + '/valid.csv'),
            '--smiles_columns', 'smiles',
            '--target_columns', 'target',
            '--dataset_type', 'multiclass',
            '--multiclass_num_classes', '3',
            '--save_dir', os.path.join(BASE_DIR, folder + '/model_checkpoint')
        ]
        args = chemprop.args.TrainArgs().parse_args(arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
        print(mean_score, std_score)
