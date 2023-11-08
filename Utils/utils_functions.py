import re
import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from . import utils_paths, utils_objects, utils_output
from sklearn.model_selection import train_test_split


def get_str_until_number(txt):
    return re.findall("[a-zA-Z]+", txt)[0]


def get_prediction_from_full_sentence(sentence, benchmark_data, task_type):
    try:
        sep = '[sep]' if benchmark_data else '{}:'.format(task_type.capitalize())
        pred_action_seq = re.findall("{} (.*)".format(sep), sentence)[-1]
    except:
        pred_action_seq = "None"

    return clean(pred_action_seq)


def create_sentence_for_gpt(src, sep, trg):
    prep_txt = f'<|startoftext|>{src}\n{sep} '
    if trg:
        prep_txt += f'{trg}<|endoftext|>'
    return prep_txt


def create_pdf(input_type, dicts, task_type, pddl=''):
    utils_output.output_file_name = 'Image_outputs'
    multiple_cols_fig = utils_output.create_multiple_cols(dicts, task_type)
    table_fig = utils_output.df_to_pdf(dicts, task_type)

    utils_output.create_pdf_with_img([multiple_cols_fig, table_fig],
                                     'Results_For_Type_{}'.format(input_type), task_type, pddl)


def create_datasets_from_csv(input_type, model_name):
    data_dir = '{}/{}/{}/'.format(utils_paths.csv_path, model_name, input_type)

    train_df = pd.read_csv('{}train_data.csv'.format(data_dir))
    val_seen_df = pd.read_csv('{}val_seen_data.csv'.format(data_dir))
    val_unseen_df = pd.read_csv('{}val_unseen_data.csv'.format(data_dir))

    return {'train': train_df,
            'val_seen': val_seen_df,
            'val_unseen': val_unseen_df}


def edit_batch_size(src_len, model_type='t5'):

    if model_type == 't5-small':
        if src_len < 300:
            bs = 32
        elif src_len < 560:
            bs = 32
        else:
            bs = 32

    elif model_type == 't5-base':
        if src_len < 100:
            bs = 32
        elif src_len < 200:
            bs = 16
        else:
            bs = 8

    else:
        if src_len < 100:
            bs = 16
        elif src_len < 300:
            bs = 8
        elif src_len < 400:
            bs = 4
        else:
            bs = 2

    print('New Batch Size: {}'.format(bs))
    return bs


# def edit_batch_size(src_len, model_type='t5'):
#
#     if model_type == 't5-small':
#         if src_len < 300:
#             bs = 16
#         elif src_len < 560:
#             bs = 8
#         else:
#             bs = 4
#
#     elif model_type == 't5-base':
#         if src_len < 100:
#             bs = 8
#         elif src_len < 200:
#             bs = 4
#         else:
#             bs = 2
#
#     else:
#         if src_len < 100:
#             bs = 16
#         elif src_len < 300:
#             bs = 8
#         elif src_len < 400:
#             bs = 4
#         else:
#             bs = 2
#
#     print('New Batch Size: {}'.format(bs))
#     return bs


def start_print(s):
    print('*' * 100)
    print(s)


def get_id_from_action(action, key='objectId'):
    obj = action["planner_action"][key]
    obj = extract_obj_id(obj)
    return obj


def extract_obj_id(obj):
    new_name = obj.lower()
    new_name = new_name.split('|')
    if 'sliced' not in obj.lower() and len(new_name) > 4:
        new_name[0] = new_name[4]

    if new_name[0] in ['sink', 'bathtub']:
        new_name[0] += 'basin'

    new_name = '|'.join(new_name[:4])
    return new_name


def in_or_on(parent, target_obj):
    # if parent in in_not_on:
    #     return '{} in {}'.format(target_obj, parent)
    return 'on {} {}'.format(target_obj, parent)


def json_to_dict(path):
    with open(path) as f:
        data = json.load(f)
        return data


def retrieve_all_task_jsons(mode='Train', json_file_name="traj_data.json"):
    all_task_desc = []
    desc_to_js = defaultdict(list)
    main_data_path = os.path.realpath(utils_paths.raw_data_path + mode)
    all_tasks_names = os.listdir(main_data_path)

    for task_desc in tqdm(all_tasks_names):
        all_task_desc.append(task_desc)
        this_task_full_path = os.path.join(main_data_path, task_desc)
        trials_names = os.listdir(this_task_full_path)

        if 'test' in mode.lower():
            json_path = os.path.join(this_task_full_path, json_file_name)
            desc_to_js[task_desc].append(json_to_dict(json_path))
            continue

        for trial_name in trials_names:
            json_path = os.path.join(this_task_full_path, trial_name, json_file_name)
            desc_to_js[task_desc].append(json_to_dict(json_path))

    return desc_to_js


def trim_spaces(small_list):
    new_small_list = []
    for s in small_list:
        if len(s) > 0:
            while s[0] == ' ' and len(s) > 0:
                s = s[1:]
                if len(s) == 0:
                    break
            if len(s) > 0:
                while s[-1] == ' ' and len(s) > 0:
                    s = s[:-1]
                    if len(s) == 0:
                        break
        new_small_list.append(s.lower())

    return new_small_list


def split_predicate_seq(seq):
    split_seq = trim_spaces(re.split("[.,]", seq))
    if split_seq[-1] == '':
        split_seq = split_seq[:-1]
    return split_seq


def check_for_duplicate_predicate(goal_predicates):
    goal_predicates = split_predicate_seq(goal_predicates)
    predicate_set = set()
    for predicate in goal_predicates:
        regex = "\\s*[()]\\s*"
        match = re.split(regex, predicate)
        for pred in match:
            if pred != '':
                predicate_set.add(f'({pred})')

    diff_list = [x for x in goal_predicates if x not in predicate_set] + \
                [x for x in predicate_set if x not in goal_predicates]

    if len(diff_list) > 0:
        print('goal_predicates: {}\npredicate_set: {}'.format(goal_predicates, predicate_set))
        print('Diff: {}'.format(diff_list))
    return predicate_set


def add_parenthesis_if_needed(predicate):
    if not predicate.startswith('('):
        predicate = '(' + predicate
    if not predicate.endswith(')'):
        predicate = predicate + ')'

    return predicate


def empty_directories():
    for path in [utils_paths.problem_path, utils_paths.valid_solutions_path, utils_paths.non_valid_solutions_path]:
        file_list = [f for f in os.listdir(path)]
        for file in file_list:
            os.remove(os.path.join(path, file))


def remove_self_duplicates(df):
    df_group_by = df.groupby(by=['col_for_duplicates'])
    idx = [x[0] for x in df_group_by.groups.values() if len(x) == 1]
    print('total unique indexes: {}\n'.format(len(idx)))
    df = df.reindex(idx)
    return df


def find_duplicates(df_dict, mode1, mode2):
    df1 = df_dict[mode1]
    df2 = df_dict[mode2]
    df = pd.concat([df1, df2])
    df = df.reset_index(drop=True)
    df_group_by = df.groupby(by=['col_for_duplicates'])
    idx = [x[0] for x in df_group_by.groups.values() if len(x) > 1]
    print('total duplicates between {} and {}: {}'.format(mode1, mode2, len(idx)))
    df = df.reindex(idx)
    return df['col_for_duplicates'].tolist()


def remove_values_from_df(df, duplicates):
    print('\ndf shape: {}'.format(df.shape))
    print('duplicates length: {}'.format(len(duplicates)))

    new_df = df[~df['col_for_duplicates'].isin(duplicates)]

    print('new_df shape: {}'.format(new_df.shape))
    return new_df


def clean_val_for_preprocess(val):
    val = val.lower()
    if len(val) > 0:
        while val[0] in ['\n', '\b', ' ', '\t', '.', ',', ';', ')', '(', '?']:
            val = val[1:]
            if len(val) == 0:
                break
    if len(val) > 0:
        while val[-1] in ['\n', '\b', ' ', '\t', '.', ',', ';', ')', '(', '?']:
            val = val[:-1]
            if len(val) == 0:
                break
    return val


def clean(val):
    del_w = utils_objects.del_w

    if '[sep]' in val:
        val = val.split('[sep] ')[1]
        while val[0] == ' ':
            val = val[1:]
    val = val.replace(' , ', ', ')
    for w in del_w:
        val = val.replace(w, ' ')

    if val:
        while val[0] == ' ':
            val = val[1:]
            if not val:
                break
            print(val)

    if val:
        if val.split(' ')[0] == 'pick':
            val = 'gto arg1, ' + val

    # for x, y in utils_objects.common_two_words_objects_gpt.items():
    #     val = val.replace(x, y)
    return val


def check_and_create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_clean_datasets(datasets, input_type, model_name, modes, train_size):
    # col_for_duplicates = input_type if input_type == 'relations_meta_wo_ids' else 'task'
    col_for_duplicates = 'task'
    self_unique = {}
    new_datasets = {}

    for mode in modes:
        datasets[mode]['col_for_duplicates'] = \
            datasets[mode][col_for_duplicates].apply(lambda x: clean_val_for_preprocess(x))

        print('Mode: {}, Length: {}'.format(mode, datasets[mode].shape[0]))
        self_unique[mode] = remove_self_duplicates(datasets[mode]).reset_index()

    for i, mode in enumerate(modes):
        for j in range(i + 1, len(modes)):
            duplicates = find_duplicates(self_unique, modes[i], modes[j])
            print('Removing duplicates between {} and {}'.format(modes[i], modes[j]))
            self_unique[modes[i]] = remove_values_from_df(self_unique[modes[i]], duplicates)
        new_datasets[modes[i]] = self_unique[modes[i]]

    print('\nChecking after cleaning...')
    for i, mode in enumerate(modes):
        for j in range(i + 1, len(modes)):
            find_duplicates(new_datasets, modes[i], modes[j])

    total_shape = 0
    for mode, df in new_datasets.items():
        total_shape += df.shape[0]

    for mode, df in new_datasets.items():
        print('\nMode {}, Shape: {}/{} = {}'.format(mode, df.shape, total_shape, round(df.shape[0]/total_shape, 2)))
        data_dir = '{}/{}/{}/{}/'.format(utils_paths.csv_path, model_name, input_type, train_size)
        df.to_csv('{}{}_clean_data.csv'.format(data_dir, mode), index=False)

    return new_datasets


def train_val_test_split(datasets, val_test_size):
    train, val = train_test_split(datasets['train'], test_size=val_test_size)
    val, test = train_test_split(val, test_size=0.5)
    new_data_dict = {'train': train,
                     'val': pd.concat([val, datasets['val_seen']]),
                     'test': test,
                     'test_unseen': datasets['val_unseen']}

    return new_data_dict


def create_object_type_dict(obj_list):
    objects = split_predicate_seq(obj_list)
    objects.sort()
    obj_type = defaultdict(list)
    for obj in objects:
        obj_type[get_str_until_number(obj)].append(obj)
    return obj_type


def valid_predicate(predicate):
    predicate_split = predicate.split(' ')
    predicate_type = predicate_split[0]
    obj1 = predicate_split[1] if len(predicate_split) > 1 else ''
    obj2 = predicate_split[2] if len(predicate_split) > 2 else ''

    if predicate_type not in utils_objects.pdf_order_dict['goal']['per_value']:
        return False

    elif predicate_type == 'on':
        if len(predicate_split) != 3 or obj1 not in utils_objects.all_obj_from_ai2thor or \
                obj2 not in utils_objects.all_obj_from_ai2thor:
            return False

    elif predicate != 'two_task':
        if len(predicate_split) != 2 or obj1 not in utils_objects.all_obj_from_ai2thor:
            return False

    return True
