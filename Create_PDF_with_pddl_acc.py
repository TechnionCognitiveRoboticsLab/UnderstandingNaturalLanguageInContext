import os
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from training import calc_pddl_correct
from Utils import utils_paths, utils_functions

models_names = ['t5-base', 'gpt2-medium']
# modes = ['val', 'test', 'test_unseen']
modes = ['test_unseen']

inputs = ['task',
          'task,relations_meta_wo_ids']
#
# inputs = ['task',
#           'task,plan',
#           'task,relations_meta_wo_ids',
#           'task,plan,relations_meta_wo_ids',
#           'relations_meta_wo_ids']

train_sizes = ['0.01', '0.05', '0.1', '0.2', '0.5', '0.75', '0.9']
# train_sizes = ['0.5']

# Before running this code, make sure that yo have the following files updated in your PC:
# train\val_seen\val_unseen_precision\recall for each model and each input type
# Beam data - full data with predictions


def csv_to_dict(mode, input_name, model_name, eval_type, both, train_size):
    data_dir = '{}/{}/{}/{}/{}/'.format(utils_paths.results_path_csv,
                                        model_name,
                                        both + 'Actions',
                                        input_name,
                                        train_size)

    tmp = pd.read_csv('{}{}_{}.csv'.format(data_dir, mode, eval_type))
    if 'Unnamed: 0' in tmp.columns:
        tmp = tmp.drop('Unnamed: 0', axis=1)
    return tmp.to_dict('records')[0]


def dict_to_csv(data_dict, mode, input_name, model_name, eval_type, both, train_size):
    data_dir = '{}{}/{}/{}/'.format(utils_paths.results_path_csv,
                                    model_name,
                                    both + 'Actions',
                                    input_name,
                                    train_size)

    pd.DataFrame(data_dict, index=[0]).to_csv('{}{}_{}.csv'.format(data_dir, mode, eval_type), index=False)


def recreate_pdf(input_name, model_name, num_epochs, train_size, both=''):
    precision = {}
    recall = {}
    precision_per_action = {}
    recall_per_action = {}

    pddl_corrects_orig_goal = {}
    pddl_corrects_pred_goal = {}

    actions_data_dir = '{}{}/{}/{}/{}/{}/'.format(utils_paths.results_path_csv, model_name, both + 'Actions',
                                                  input_name, train_size, 'Beam_Data')

    goal_data_dir = '{}{}/{}/{}/{}/{}/'.format(utils_paths.results_path_csv, model_name, both + 'Goal',
                                               input_name, train_size, 'Beam_Data')

    clean_data_dir = '{}{}/{}/{}/'.format(utils_paths.csv_path, model_name, input_name, train_size)

    for mode in modes:
        clean_df = pd.read_csv('{}{}_clean_data.csv'.format(clean_data_dir, mode))
        actions_df = pd.read_csv('{}{}_data_with_beam_predictions.csv'.format(actions_data_dir, mode))
        goal_df = pd.read_csv('{}{}_data_with_beam_predictions.csv'.format(goal_data_dir, mode))

        on_col = 'task' if input_name == 'relations_meta_wo_ids' else 'input_text'
        df = actions_df.merge(goal_df, how='inner', on=[on_col], suffixes=('_actions', '_goal'))
        if df.shape[0] != actions_df.shape[0]:
            print('Merge result has a different shape!')
            continue

        df = df.merge(clean_df, how='inner', on=[on_col], suffixes=('', '_clean'))
        if df.shape[0] != actions_df.shape[0]:
            print('Merge result has a different shape after clean merge!')
            continue

        df['pred_goal_first'] = df['pred_goal_beam'].apply(lambda x: str(x).split('|')[0])

        pddl_correct_orig_goal = calc_pddl_correct(df['orig_actions'], df['pred_actions_beam'], df, mode, model_name,
                                                   input_name, 'orig_goal', both)

        pddl_corrects_orig_goal[mode] = round(pddl_correct_orig_goal/df.shape[0], 2)

        pddl_correct_pred_goal = calc_pddl_correct(df['orig_actions'], df['pred_actions_beam'], df, mode, model_name,
                                                   input_name, 'pred_goal_first', both)

        pddl_corrects_pred_goal[mode] = round(pddl_correct_pred_goal/df.shape[0], 2)

        precision[mode] = csv_to_dict(mode, input_name, model_name, 'precision', both, train_size)
        precision[mode]['full_pddl'] = pddl_corrects_orig_goal[mode]
        precision[mode]['full_pddl_pred_goal'] = pddl_corrects_pred_goal[mode]
        dict_to_csv(precision[mode], mode, input_name, model_name, 'precision', both, train_size)

        recall[mode] = csv_to_dict(mode, input_name, model_name, 'recall', both, train_size)
        recall[mode]['full_pddl'] = pddl_corrects_orig_goal[mode]
        recall[mode]['full_pddl_pred_goal'] = pddl_corrects_pred_goal[mode]
        dict_to_csv(recall[mode], mode, input_name, model_name, 'recall', both, train_size)

        precision_per_action[mode] = csv_to_dict(mode, input_name, model_name,
                                                 'precision_per_action', both, train_size)
        recall_per_action[mode] = csv_to_dict(mode, input_name, model_name,
                                              'recall_per_action', both, train_size)

    dicts = {'Precision_General': precision,
             'Recall_General': recall,
             'Precision_per_value': precision_per_action,
             'Recall_per_value': recall_per_action}

    pdf_string = 'AFTER-PDDL_{}_Model-{}{}_Epochs-{}_Train_Split-{}'.format(input_name, model_name,
                                                                            both, num_epochs, train_size)

    utils_functions.create_pdf(pdf_string, dicts, 'actions', pddl='PDDL/')


def create_full_type_df_actions(train_size, both=False):
    gen_keys = ['command', 'arg1', 'p_arg1', 'arg2', 'p_arg2', 'full_triple', 'full_seq', 'full_minus1', 'full_pddl',
                'full_pddl_pred_goal']

    methods = ['recall', 'precision']

    gen_dic = {key: defaultdict(float) for key in gen_keys}
    method_dic = {key: deepcopy(gen_dic) for key in methods}
    mode_dic = {key: deepcopy(method_dic) for key in modes}
    model_dic = {m: deepcopy(mode_dic) for m in models_names}

    for model_name in models_names:
        if 't5' in model_name and both:
            path = '{}{}/Both_Actions'.format(utils_paths.results_path_csv, model_name)
        else:
            path = '{}{}/Actions'.format(utils_paths.results_path_csv, model_name)
        input_names = os.listdir(path)
        input_names.sort()

        for mode in modes:
            for input_name in input_names:
                if 'All' in input_name:
                    continue
                for method in methods:
                    file = '{}/{}/{}/{}_{}.csv'.format(path, input_name, train_size, mode, method)
                    temp_df = pd.read_csv(file)
                    for key in gen_dic:
                        model_dic[model_name][mode][method][key][input_name] = temp_df[key][0]

    for mode in modes:
        for eval_type in methods:
            for model_name, all_modes in model_dic.items():
                print('\n\nModel: {}\nMode: {}\nEvaluation Method: {}'.format(model_name, mode, eval_type))
                df = pd.DataFrame(all_modes[mode][eval_type]).sort_index(key=lambda x: x.str.len())
                if 't5' in model_name and both:
                    dir_path = '{}{}/Both_Actions/'.format(utils_paths.results_path_csv, model_name)
                else:
                    dir_path = '{}{}/Actions/'.format(utils_paths.results_path_csv, model_name)

                df.to_csv('{}{}/All_type_{}_precision.csv'.format(dir_path, train_size, mode))


def create_full_type_df_goal(train_size, both=False):
    gen_keys = ['predicate_type', 'arg1', 'p_arg1', 'arg2', 'p_arg2',
                'f_predicate', 'f_predicate_sim', 'f_seq', 'f_seq_sim']

    methods = ['recall', 'precision']

    gen_dic = {key: defaultdict(float) for key in gen_keys}
    method_dic = {key: deepcopy(gen_dic) for key in methods}
    mode_dic = {key: deepcopy(method_dic) for key in modes}
    model_dic = {m: deepcopy(mode_dic) for m in models_names}

    for model_name in models_names:
        if 't5' in model_name and both:
            path = '{}{}/Both_Goal'.format(utils_paths.results_path_csv, model_name)
        else:
            path = '{}{}/Goal'.format(utils_paths.results_path_csv, model_name)
        input_names = os.listdir(path)
        input_names.sort()

        for mode in modes:
            for input_name in input_names:
                if 'All' in input_name:
                    continue
                for method in methods:
                    file = '{}/{}/{}/{}_{}.csv'.format(path, input_name, train_size, mode, method)
                    temp_df = pd.read_csv(file)
                    for key in gen_dic:
                        model_dic[model_name][mode][method][key][input_name] = temp_df[key][0]

    for mode in modes:
        for eval_type in methods:
            for model_name, all_modes in model_dic.items():
                print('\n\nModel: {}\nMode: {}\nEvaluation Method: {}'.format(model_name, mode, eval_type))
                df = pd.DataFrame(all_modes[mode][eval_type]).sort_index(key=lambda x: x.str.len())
                if 't5' in model_name and both:
                    dir_path = '{}{}/Both_Goal/'.format(utils_paths.results_path_csv, model_name)
                else:
                    dir_path = '{}{}/Goal/'.format(utils_paths.results_path_csv, model_name)

                df.to_csv('{}{}/All_type_{}_precision.csv'.format(dir_path, train_size, mode))


for input_ in inputs:
    for size in train_sizes:
        # print('\nRunning PDDL predictions for model {} and input type {} and train size: {}'
        #       .format('gpt2-medium', input_, size))
        # recreate_pdf(input_, 'gpt2-medium', 25, size, '')
        print('\nRunning PDDL predictions for model {} and input type {} and train size: {}'
              .format('t5-base', input_, size))
        recreate_pdf(input_, 't5-base', 25, size, 'Both_')


# create_full_type_df_actions(True)
# create_full_type_df_goal(True)
