import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from data_validator import validate_orig_data
from Utils import utils_model, utils_variables, utils_paths, utils_functions


def creat_input_col(df, columns):
    col_name_dict = {'relations': 'Relations',
                     'relations_meta': 'Relations',
                     'relations_meta_wo_ids': 'Relations',
                     'objects': 'Objects',
                     'obj_meta': 'Objects',
                     'task': 'Task',
                     'plan': 'Plan'}

    input_col_name = 'input_text'
    columns = columns.split(',')
    df[input_col_name] = ''
    tqdm_obj = tqdm(df.iterrows(), total=df.shape[0])

    for index, row in tqdm_obj:
        this_row_input = []
        for col in columns:

            this_row_input.append('{}: {}'.format(col_name_dict[col],
                                                  utils_functions.clean_val_for_preprocess(row[col])))

        row[input_col_name] = ', '.join(this_row_input)
        df.at[index] = row

    return df


def create_df_for_prediction(orig_df, input_cols):
    new_df = creat_input_col(deepcopy(orig_df), input_cols)
    for task_type in ['goal', 'actions']:
        col = task_type
        new_df.rename(columns={col: "target_text_{}".format(task_type)}, inplace=True)
        new_df['prefix_{}'.format(task_type)] = utils_variables.prefixes[task_type]
    return new_df


def validate_and_split(args, train_size):
    if args.validate_data:
        for mode in ['val/seen', 'val/unseen']:  # 'train',
            validate_orig_data(mode)

    data_path = '{}{}/'.format(utils_paths.csv_path, train_size)
    utils_paths.check_and_create_dir(data_path)
    datasets = {}
    for mode in ['orig_train', 'val_seen', 'val_unseen']:
        mode_to_dict = 'train' if 'train' in mode else mode
        datasets[mode_to_dict] = pd.read_csv('{}{}_data_with_pddl.csv'.format(utils_paths.csv_path, mode))

    if args.re_split:
        new_datasets = utils_functions.train_val_test_split(datasets, args.train_test_split)
    else:
        new_datasets = {'train': datasets['train'],
                        'val': datasets['val_seen'],
                        'test': datasets['val_unseen'],
                        'test_unseen': datasets['val_unseen']}

    for mode, df in new_datasets.items():
        print('\nMode {}, Shape: {}'.format(mode, df.shape))
        df.to_csv('{}{}_data_with_pddl.csv'.format(data_path, mode), index=False)


def read_data(mode, args, train_size):
    print("Creating big file data, mode: {}".format(mode.capitalize()))
    data_path = '{}{}/'.format(utils_paths.csv_path, train_size)

    df = pd.read_csv('{}{}_data_with_pddl.csv'.format(data_path, mode))
    df = df[df['solution valid'] == 'Valid']

    prediction_df = create_df_for_prediction(deepcopy(df), args.input_type)
    data_dir = '{}/{}/{}/{}/'.format(utils_paths.csv_path, args.model_type, args.input_type, train_size)
    utils_paths.check_and_create_dir(data_dir)

    prediction_df.to_csv('{}{}_data.csv'.format(data_dir, mode), index=False)
    return prediction_df


def find_lengths(train_df, task_type, tokenizer):
    max_src_len = 0
    max_trg_len = 0

    tqdm_obj = tqdm(train_df.iterrows(), total=len(train_df))
    for index, row in tqdm_obj:
        prefix = row['prefix_{}'.format(task_type)]
        src = f'{prefix} {row["input_text"]}'
        trg = row['target_text_{}'.format(task_type)]

        src_len = len(tokenizer(src).data['input_ids'])
        trg_len = len(tokenizer(trg).data['input_ids'])

        max_src_len = max(max_src_len, src_len)
        max_trg_len = max(max_trg_len, trg_len)

    utils_model.model_params[task_type]['max_src_len'] = min(max_src_len, 1024)
    utils_model.model_params[task_type]['max_trg_len'] = max_trg_len
    print('Max src length: {} \nMax trg length: {}'.format(max_src_len, max_trg_len))
    return min(max_src_len, 1024), max_trg_len


def create_big_df(data_dir, mode):
    mode_df = pd.read_csv('{}{}_clean_data.csv'.format(data_dir, mode))

    goal_df = mode_df[['prefix_goal', 'input_text', 'target_text_goal', 'task']]
    goal_df = goal_df.rename(columns={'prefix_goal': 'prefix_both', 'target_text_goal': 'target_text_both'})

    actions_df = mode_df[['prefix_actions', 'input_text', 'target_text_actions', 'task']]
    actions_df = actions_df.rename(columns={'prefix_actions': 'prefix_both', 'target_text_actions': 'target_text_both'})

    df = goal_df.append(actions_df, ignore_index=True)
    df.to_csv('{}Both_{}_clean_data.csv'.format(data_dir, mode), index=False)
    return df


def load_clean_datasets(model_name, input_type, modes, train_size):
    datasets = {}
    data_dir = '{}/{}/{}/'.format(utils_paths.csv_path, model_name, input_type, train_size)
    for mode in modes:
        datasets[mode] = pd.read_csv('{}{}_clean_data.csv'.format(data_dir, mode))
    return datasets
