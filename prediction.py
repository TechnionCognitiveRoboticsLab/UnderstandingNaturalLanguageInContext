import re
import training
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import torch.multiprocessing
from collections import defaultdict
from Utils import utils_model, utils_variables, utils_functions, utils_paths, utils_objects

torch.multiprocessing.set_sharing_strategy('file_system')


def get_ac_type(action_type):
    if 'go to' in action_type:
        return 'goto'
    elif 'pick up' in action_type:
        return 'pickup'

    action_type = trim_symbols_from_pddl_goal(action_type)
    return action_type.split(' ')[0]


def trim_symbols_from_pddl_goal(small_predicate):
    if small_predicate == '':
        return ''
    while small_predicate[0] in [',', '(', ')', ' ']:
        small_predicate = small_predicate[1:]
        if small_predicate == '':
            return ''
    while small_predicate[-1] in [',', '(', ')', ' ']:
        small_predicate = small_predicate[:-1]
        if small_predicate == '':
            return ''
    return small_predicate.lower()


def calc_total_triples(action_list, total_counter, mode):
    counter = 0
    per_action_total = defaultdict(int)

    for action_seq in action_list:
        split_actions = action_seq.split('.')
        if split_actions[-1] == '':
            split_actions = split_actions[:-1]
        counter += len(split_actions)

        for small_action in split_actions:
            per_action_total[get_ac_type(small_action)] += 1

    total_counter['triples_{}'.format(mode)] = counter

    for key, val in per_action_total.items():
        total_counter['{}_{}'.format(key, mode)] = val

    return total_counter


def get_all_args_from_action(action):
    arg2_exist = False
    action = trim_symbols_from_pddl_goal(action)
    command = get_ac_type(action)

    for long_name, short_name in utils_objects.common_two_words_objects.items():
        if long_name in action:
            action = action.replace(long_name, short_name)

    action_split = action.split(' ')
    if '' in action_split:
        action_split.remove('')

    arg1 = ''
    if len(action_split) > 1:
        arg1 = action_split[1]
    arg2 = ''
    if len(action_split) > 2:
        arg2 = action_split[2]
        arg2_exist = True

    return {'command': command, 'arg1': arg1, 'arg2': arg2}, arg2_exist


def update_confusion_matrix(confusion_matrix, pred_args, true_args):
    for arg in ['arg1', 'arg2']:
        prd = pred_args[arg]
        tru = true_args[arg]
        if prd == '' or tru == '':
            continue
        if prd != tru:
            if prd not in confusion_matrix:
                confusion_matrix[prd] = defaultdict(int)
            confusion_matrix[prd][tru] += 1

    return confusion_matrix


def calc_all_triple_acc(pred_action,
                        true_action,
                        correct_counter_dict,
                        per_action_correct,
                        total_counter,
                        confusion_matrix):
    full_sequence_equal = True

    if pred_action == '' or true_action == '':
        if pred_action == '' and true_action == '':
            return correct_counter_dict, per_action_correct, total_counter, False, confusion_matrix
        return correct_counter_dict, per_action_correct, total_counter, True, confusion_matrix

    pred_args, arg2_exist = get_all_args_from_action(pred_action)
    if arg2_exist:
        total_counter['arg2_pred'] += 1

    true_args, arg2_exist = get_all_args_from_action(true_action)
    if arg2_exist:
        total_counter['arg2_true'] += 1

    confusion_matrix = update_confusion_matrix(confusion_matrix, pred_args, true_args)

    if pred_action == true_action:
        correct_counter_dict['full_triple'] += 1
        per_action_correct[pred_args['command']] += 1
    else:
        full_sequence_equal = False

    for arg_name in pred_args:
        pred_val = pred_args[arg_name]
        true_val = true_args[arg_name]

        if pred_val == true_val:
            if pred_val != '':
                correct_counter_dict[arg_name] += 1
                if 'arg' in arg_name:
                    correct_counter_dict['p_{}'.format(arg_name)] += 1
        else:
            if pred_val in utils_objects.similar_objects and utils_objects.similar_objects[pred_val] == true_val or \
                    true_val in utils_objects.similar_objects and utils_objects.similar_objects[true_val] == pred_val:
                correct_counter_dict['p_{}'.format(arg_name)] += 1

    return correct_counter_dict, per_action_correct, total_counter, full_sequence_equal, confusion_matrix


def create_precision_and_recall_dicts(correct_counter_dict, per_action_correct, total_counter, pddl_correct):
    precision_dict = {'arg2': round(correct_counter_dict['arg2'] / total_counter['arg2_pred'], 3),
                      'p_arg2': round(correct_counter_dict['p_arg2'] / total_counter['arg2_pred'], 3),

                      'full_seq': round(correct_counter_dict['full_seq'] / total_counter['sequences'], 3),
                      'full_pddl': round(pddl_correct / total_counter['sequences'], 3),
                      'full_pddl_pred_goal': round(0 / total_counter['sequences'], 3),
                      'full_minus1': round(correct_counter_dict['full_minus1'] /
                                           total_counter['sequences'], 3)
                      }

    recall_dict = {'arg2': round(correct_counter_dict['arg2'] / total_counter['arg2_true'], 3),
                   'p_arg2': round(correct_counter_dict['p_arg2'] / total_counter['arg2_true'], 3),

                   'full_seq': round(correct_counter_dict['full_seq'] / total_counter['sequences'], 3),
                   'full_pddl': round(pddl_correct / total_counter['sequences'], 3),
                   'full_pddl_pred_goal': round(0 / total_counter['sequences'], 3),
                   'full_minus1': round(correct_counter_dict['full_minus1'] /
                                        total_counter['sequences'], 3)}

    precision_per_action = {}
    recall_per_action = {}
    for correct_type, val in per_action_correct.items():
        pred_name = correct_type + '_pred'
        true_name = correct_type + '_true'
        if pred_name in total_counter:
            precision_per_action[correct_type] = round(val / total_counter[pred_name], 3)
            recall_per_action[correct_type] = round(val / total_counter[true_name], 3)

    for key in ['command', 'arg1', 'p_arg1', 'full_triple']:
        precision_dict[key] = round(correct_counter_dict[key] / total_counter['triples_pred'], 3)
        recall_dict[key] = round(correct_counter_dict[key] / total_counter['triples_true'], 3)

    return {'precision': precision_dict,
            'precision_per_action': precision_per_action,
            'recall': recall_dict,
            'recall_per_action': recall_per_action}


def accuracy_cal_actions(orig_actions, pred_actions, mode, input_name, train_size,
                         model_name, pddl_correct, both_add=''):
    per_action_correct = defaultdict(int)
    correct_counter_dict = deepcopy(utils_objects.action_correct_dict)
    total_counter = deepcopy(utils_objects.total_counter)

    total_counter = calc_total_triples(pred_actions, total_counter, 'pred')
    total_counter = calc_total_triples(orig_actions, total_counter, 'true')

    total_counter['sequences'] = len(orig_actions)

    confusion_matrix = {}

    for j, y in enumerate(orig_actions):
        full_sequence_equal = True
        full_sequence_correct_wo_first_triple = True

        y = utils_functions.split_predicate_seq(y)
        y_pred = utils_functions.split_predicate_seq(pred_actions[j])

        for i in range(min(len(y_pred), len(y))):
            pred_action = y_pred[i]
            true_action = y[i]

            correct_counter_dict, per_action_correct, total_counter, triple_equal, confusion_matrix = \
                calc_all_triple_acc(pred_action,
                                    true_action,
                                    correct_counter_dict,
                                    per_action_correct,
                                    total_counter,
                                    confusion_matrix)

            if not triple_equal:
                full_sequence_equal = False
                if i > 0:
                    full_sequence_correct_wo_first_triple = False

        if full_sequence_equal:
            correct_counter_dict['full_seq'] += 1

        if full_sequence_correct_wo_first_triple:
            correct_counter_dict['full_minus1'] += 1

    precision_recall_dicts = create_precision_and_recall_dicts(correct_counter_dict, per_action_correct,
                                                               total_counter, pddl_correct)

    data_dir = '{}/{}/{}/{}/{}/'.format(utils_paths.results_path_csv, model_name,
                                        both_add + 'Actions', input_name, train_size)
    utils_paths.check_and_create_dir(data_dir)

    for name, dic in precision_recall_dicts.items():
        pd.DataFrame(dic, index=[0]).to_csv('{}{}_{}.csv'.format(data_dir, mode, name))

    return precision_recall_dicts


def add_params_from_predicate(small_predicate, pred_true, total_counter_dict, total_counter_dict_object):
    predicate_type = small_predicate[0]
    predicate_obj1 = small_predicate[1] if len(small_predicate) >= 2 else ''
    predicate_obj2 = small_predicate[2] if len(small_predicate) == 3 else ''

    if predicate_type != '':
        name = '{}_{}'.format(predicate_type, pred_true)
        if name in total_counter_dict:
            total_counter_dict[name] += 1

    for i, obj in enumerate([predicate_obj1, predicate_obj2]):
        if obj != '':
            total_counter_dict['arg{}_{}'.format(i + 1, pred_true)] += 1
            total_counter_dict_object['{}_{}'.format(obj, pred_true)] += 1

    return total_counter_dict, total_counter_dict_object


def calc_total_num_for_predicates(goal_predicates_list, total_counter_dict, total_counter_dict_object, pred_true):
    total_counter_dict['sequences'] = len(goal_predicates_list)

    for goal_predicate in goal_predicates_list:
        split_predicate = re.split("[.,]", goal_predicate)

        if split_predicate[-1] == '':
            split_predicate = split_predicate[:-1]

        total_counter_dict['predicates_{}'.format(pred_true)] += len(split_predicate)

        for small_predicate in split_predicate:
            small_predicate = trim_symbols_from_pddl_goal(small_predicate).split(' ')
            total_counter_dict, total_counter_dict_object = \
                add_params_from_predicate(small_predicate, pred_true, total_counter_dict, total_counter_dict_object)

    return total_counter_dict, total_counter_dict_object


def check_for_similar_predicate(predicate_params, y):
    predicate_type = predicate_params['predicate_type']
    predicate_obj_1 = predicate_params['predicate_obj_1']
    predicate_obj_2 = predicate_params['predicate_obj_2']

    similar_obj_1_list = ['']
    similar_obj_2_list = ['']

    if predicate_obj_1 in utils_objects.similar_objects:
        similar_obj_1_list = utils_objects.similar_objects[predicate_obj_1]

    if predicate_obj_2 in utils_objects.similar_objects:
        similar_obj_2_list = utils_objects.similar_objects[predicate_obj_2]

    candidates = [true_predicate for true_predicate in y if
                  trim_symbols_from_pddl_goal(true_predicate).split(' ')[0] == predicate_type]

    combinations = []

    if similar_obj_1_list != [''] or similar_obj_2_list != ['']:
        for similar_obj_1 in similar_obj_1_list:
            for similar_obj_2 in similar_obj_2_list:
                combinations += [{'({} {})'.format(predicate_type, similar_obj_1):
                                      ['predicate_type', 'p_arg1'],
                                  '({} {} {})'.format(predicate_type, predicate_obj_1, similar_obj_2):
                                      ['predicate_type', 'arg1', 'p_arg1', 'p_arg2'],
                                  '({} {} {})'.format(predicate_type, similar_obj_1, predicate_obj_2):
                                      ['predicate_type', 'p_arg1', 'p_arg2', 'arg2'],
                                  '({} {} {})'.format(predicate_type, similar_obj_1, similar_obj_2):
                                      ['predicate_type', 'p_arg1', 'p_arg2']}]

    for comb_dic in combinations:
        for val, list_name in comb_dic.items():
            if val in candidates:
                return True, list_name

    return False, -1


def get_params_from_predicate(goal_predicate):
    goal_predicate = trim_symbols_from_pddl_goal(goal_predicate)
    predicate_split = goal_predicate.split(' ')

    params = {'predicate_type': predicate_split[0] if len(predicate_split) > 0 else '',
              'predicate_obj_1': predicate_split[1] if len(predicate_split) > 1 else '',
              'predicate_obj_2': predicate_split[2] if len(predicate_split) > 2 else ''}

    return params


def create_precision_and_recall_dicts_goal(correct_counter_dict,
                                           per_goal_correct,
                                           total_counter):
    precision_dict = \
        {'arg1': round(correct_counter_dict['arg1'] / total_counter['arg1_pred'], 3),
         'p_arg1': round(correct_counter_dict['p_arg1'] / total_counter['arg1_pred'], 3),

         'arg2': round(correct_counter_dict['arg2'] / total_counter['arg2_pred'], 3),
         'p_arg2': round(correct_counter_dict['p_arg2'] / total_counter['arg2_pred'], 3),

         'f_seq': round(correct_counter_dict['f_seq'] / total_counter['sequences'], 3),
         'f_seq_sim': round(correct_counter_dict['f_seq_sim'] / total_counter['sequences'], 3)}

    recall_dict = \
        {'arg1': round(correct_counter_dict['arg1'] / total_counter['arg1_true'], 3),
         'p_arg1': round(correct_counter_dict['p_arg1'] / total_counter['arg1_true'], 3),

         'arg2': round(correct_counter_dict['arg2'] / total_counter['arg2_true'], 3),
         'p_arg2': round(correct_counter_dict['p_arg2'] / total_counter['arg2_true'], 3),

         'f_seq': round(correct_counter_dict['f_seq'] / total_counter['sequences'], 3),
         'f_seq_sim': round(correct_counter_dict['f_seq_sim'] / total_counter['sequences'], 3)}

    precision_per_predicate_type = {}
    recall_per_predicate_type = {}
    for correct_type, val in per_goal_correct.items():
        pred_name = correct_type + '_pred'
        true_name = correct_type + '_true'
        if pred_name in total_counter:
            precision_per_predicate_type[correct_type] = round(val / total_counter[pred_name], 3)
            recall_per_predicate_type[correct_type] = round(val / total_counter[true_name], 3)

    precision_per_obj = {}
    recall_per_obj = {}
    for key in ['predicate_type', 'f_predicate', 'f_predicate_sim']:
        precision_dict[key] = round(correct_counter_dict[key] / total_counter['predicates_pred'], 3)
        recall_dict[key] = round(correct_counter_dict[key] / total_counter['predicates_true'], 3)
    
    return {'precision': precision_dict, 
            'precision_per_predicate_type': precision_per_predicate_type,
            'precision_per_obj': precision_per_obj,

            'recall': recall_dict,
            'recall_per_predicate_type': recall_per_predicate_type,
            'recall_per_obj': recall_per_obj}


def add_score_predicate_exist(correct_counter_dict, per_goal_correct, per_obj_correct, predicate_params):
    correct_counter_dict['f_predicate'] += 1
    correct_counter_dict['f_predicate_sim'] += 1
    correct_counter_dict['predicate_type'] += 1

    per_goal_correct[predicate_params['predicate_type']] += 1

    for i in range(1, 3):
        if predicate_params[f'predicate_obj_{i}'] != '':
            correct_counter_dict[f'arg{i}'] += 1
            correct_counter_dict[f'p_arg{i}'] += 1
            per_obj_correct[predicate_params[f'predicate_obj_{i}']] += 1

    return correct_counter_dict, per_goal_correct, per_obj_correct


def accuracy_cal_goals(orig_goal, pred_goal, mode, input_name, train_size, model_name, both_add=''):
    correct_counter_dict = deepcopy(utils_objects.goal_correct_dict)
    total_counter_dict = deepcopy(utils_objects.total_counter_goal)
    total_counter_dict_object = defaultdict(int)

    total_counter_dict, total_counter_dict_object = \
        calc_total_num_for_predicates(pred_goal, total_counter_dict, total_counter_dict_object, 'pred')
    total_counter_dict, total_counter_dict_object = \
        calc_total_num_for_predicates(orig_goal, total_counter_dict, total_counter_dict_object, 'true')

    per_goal_correct = defaultdict(int)
    per_obj_correct = defaultdict(int)

    for j, y in enumerate(orig_goal):
        full_sequence_equal = True
        full_sequence_correct_wo_similar_objects = True

        y = utils_functions.split_predicate_seq(y)
        y_pred = utils_functions.split_predicate_seq(pred_goal[j])

        predicate_set = set()

        for goal_predicate in y_pred:
            if goal_predicate in predicate_set:
                continue

            goal_predicate = utils_functions.add_parenthesis_if_needed(goal_predicate)
            predicate_params = get_params_from_predicate(goal_predicate)

            if goal_predicate in y:
                correct_counter_dict, per_goal_correct, per_obj_correct = \
                    add_score_predicate_exist(correct_counter_dict, per_goal_correct, per_obj_correct, predicate_params)

            elif 'two_task' not in goal_predicate:
                full_sequence_equal = False
                similar_exist, list_of_correct_keys = check_for_similar_predicate(predicate_params, y)
                if similar_exist:
                    correct_counter_dict['f_predicate_sim'] += 1
                    per_goal_correct[predicate_params['predicate_type']] += 1
                    for name in list_of_correct_keys:
                        correct_counter_dict[name] += 1
                else:
                    full_sequence_correct_wo_similar_objects = False

            predicate_set.add(goal_predicate)

        if full_sequence_equal:
            correct_counter_dict['f_seq'] += 1

        if full_sequence_correct_wo_similar_objects:
            correct_counter_dict['f_seq_sim'] += 1

    precision_recall_dicts = \
        create_precision_and_recall_dicts_goal(correct_counter_dict, per_goal_correct, total_counter_dict)

    data_dir = '{}/{}/{}/{}/{}/'.format(utils_paths.results_path_csv, model_name,
                                        both_add + 'Goal', input_name, train_size)
    utils_paths.check_and_create_dir(data_dir)
    
    for name, dic in precision_recall_dicts.items():
        pd.DataFrame(dic, index=[0]).to_csv('{}{}_{}.csv'.format(data_dir, mode, name))

    return precision_recall_dicts


def acc_and_print(actual, predictions_first, task_type, mode, input_name,
                  model_name, pddl_correct, both_add, train_size):
    if task_type == 'goal':
        precision_recall_dicts = accuracy_cal_goals(actual, predictions_first, mode,
                                                    input_name, train_size, model_name, both_add)

        precision_dict = precision_recall_dicts['precision']
        precision_per_predicate_type = precision_recall_dicts['precision_per_predicate_type']
        recall_dict = precision_recall_dicts['recall']
        recall_per_predicate_type = precision_recall_dicts['recall_per_predicate_type']

        print('\n{}\nMode: {}\n'
              'Avg per-goal Accuracy:\n'

              'Predicate Type Precision: {}\n'
              'Predicate Type Recall: {}\n'

              'Arg1 Precision: {}\n'
              'Arg1 Permissive Precision: {}\n'
              'Arg1 Recall: {}\n'
              'Arg1 Permissive Recall: {}\n'

              'Arg2 Precision: {}\n'
              'Arg2 Permissive Precision: {}\n'
              'Arg2 Recall: {}\n'
              'Arg2 Permissive Recall: {}\n'

              'Full Predicate Precision: {}\n'
              'Full Predicate Recall: {}\n'
              'Full Sequence: {}\n'
              'Full Similar: {}\n\n'.format('-' * 50,
                                            mode,
                                            precision_dict['predicate_type'],
                                            recall_dict['predicate_type'],

                                            precision_dict['arg1'],
                                            precision_dict['p_arg1'],
                                            recall_dict['arg1'],
                                            recall_dict['p_arg1'],

                                            precision_dict['arg2'],
                                            precision_dict['p_arg2'],
                                            recall_dict['arg2'],
                                            recall_dict['p_arg2'],

                                            precision_dict['f_predicate'],
                                            recall_dict['f_predicate'],
                                            recall_dict['f_seq'],
                                            recall_dict['f_seq_sim']))

        return precision_dict, precision_per_predicate_type, recall_dict, recall_per_predicate_type

    else:
        precision_recall_dicts = accuracy_cal_actions(actual, predictions_first, mode, input_name,
                                                      train_size, model_name, pddl_correct, both_add)

        precision_dict = precision_recall_dicts['precision']
        precision_per_action = precision_recall_dicts['precision_per_action']
        recall_dict = precision_recall_dicts['recall']
        recall_per_action = precision_recall_dicts['recall_per_action']

        print('\n{}\nMode: {}\n'
              'Avg per-action Accuracy:\n'

              'Command Precision: {}\n'
              'Command Recall: {}\n'

              'Arg1 Precision: {}\n'
              'Arg1 Permissive Precision: {}\n'
              'Arg1 Recall: {}\n'
              'Arg1 Permissive Recall: {}\n'

              'Arg2 Precision: {}\n'
              'Arg2 Permissive Precision: {}\n'
              'Arg2 Recall: {}\n'
              'Arg2 Permissive Recall: {}\n'

              'Full Triples Precision: {}\n'
              'Full Triples Recall: {}\n'
              'Full Sequence: {}\n'
              'Full Minus 1: {}\n'
              'Full With PDDL: {}\n\n'.format('-' * 50, mode,
                                              precision_dict['command'],
                                              recall_dict['command'],

                                              precision_dict['arg1'],
                                              precision_dict['p_arg1'],
                                              recall_dict['arg1'],
                                              recall_dict['p_arg1'],

                                              precision_dict['arg2'],
                                              precision_dict['p_arg2'],
                                              recall_dict['arg2'],
                                              recall_dict['p_arg2'],

                                              precision_dict['full_triple'],
                                              recall_dict['full_triple'],
                                              recall_dict['full_seq'],
                                              recall_dict['full_minus1'],
                                              recall_dict['full_pddl']))

        return precision_dict, precision_per_action, recall_dict, recall_per_action


def validate(tokenizer, model, loader, model_params, beam):
    model.eval()

    predictions = []
    predictions_first = []
    actual = []
    val_tqdm = tqdm(loader, total=len(loader))

    with torch.no_grad():
        for i, data in enumerate(val_tqdm):
            y = data['target_ids'].to(utils_model.device, dtype=torch.long)
            ids = data['source_ids'].to(utils_model.device, dtype=torch.long)
            mask = data['source_mask'].to(utils_model.device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=model_params['max_trg_len'],
                num_beams=beam,
                num_return_sequences=beam,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                     for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if utils_variables.multiple_sentences:
                preds_only_first = [preds[beam * i] for i in range(ids.shape[0])]
                preds = ['|'.join(preds[beam * i:beam * i + beam])
                         for i in range(ids.shape[0])]
                predictions_first.extend(preds_only_first)

            predictions.extend(preds)
            actual.extend(target)

            if not utils_variables.multiple_sentences:
                predictions_first = predictions

    return predictions_first, predictions, actual


def predict_from_prompt_gpt(prompt_text, tokenizer, model, beam, task_type):
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
    ids = inputs['input_ids'].cuda()
    mask = inputs['attention_mask'].cuda()

    output_sequences = model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=False,
        num_beams=beam,
        num_return_sequences=beam
    )
    beam_outputs = [utils_functions.get_prediction_from_full_sentence
                    (tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True), False, task_type)
                    for g in output_sequences]

    return beam_outputs


def predict_from_prompt_t5(prompt_text, tokenizer, model, beam):
    inputs = tokenizer(prompt_text, return_tensors="pt")

    ids = inputs['input_ids'].cuda()
    mask = inputs['attention_mask'].cuda()

    generated_ids = model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=False,
        num_beams=beam,
        num_return_sequences=beam,
        max_length=1024
    )

    beam_outputs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in generated_ids]
    return beam_outputs


def predict_from_sentences(sentences, model_name, input_type, beam):
    if 'gpt' in model_name:
        actions_dir_path = '{}/{}/{}/{}'.format(utils_paths.trained_model_path, model_name, 'Actions', input_type)
        goal_dir_path = '{}/{}/{}/{}'.format(utils_paths.trained_model_path, model_name, 'Goal', input_type)
    else:
        actions_dir_path = '{}/{}/{}/{}'.format(utils_paths.trained_model_path, model_name, 'Both', input_type)
        goal_dir_path = '{}/{}/{}/{}'.format(utils_paths.trained_model_path, model_name, 'Both', input_type)

    outputs = {'Actions': [], 'Goal': []}
    beams = {'Actions': beam, 'Goal': 1}

    for sent in sentences:
        print('\n'*3)

        for input_name in input_type.split(','):
            if '{}:'.format(input_name.capitalize()) not in sent:
                print('Could not find input {} in sentence'.format(input_name))

        if 'gpt' in model_name.lower():
            action_model, action_tokenizer = training.load_model(actions_dir_path)
            goal_model, goal_tokenizer = training.load_model(goal_dir_path)

            models = {'Actions': action_model, 'Goal': goal_model}
            tokenizers = {'Actions': action_tokenizer, 'Goal': goal_tokenizer}

            for task_type in ['Actions', 'Goal']:
                sep = '{}:'.format(task_type)
                prompt_text = utils_functions.create_sentence_for_gpt(sent, sep, '')
                outputs[task_type] = predict_from_prompt_gpt(prompt_text, tokenizers[task_type],
                                                             models[task_type], beams[task_type], task_type)

        else:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(actions_dir_path).to(utils_model.device)
            tokenizer = T5Tokenizer.from_pretrained(actions_dir_path)
            for task_type in ['actions', 'goal']:
                prompt_text = "{}: {}".format(utils_variables.prefixes[task_type], sent)
                outputs[task_type] = predict_from_prompt_t5(prompt_text, tokenizer, model,
                                                            beams[task_type.capitalize()])

        print('\nInput sentence: {}\n'.format(sent))

        print("Output:\n" + 100 * '-')
        print('Predicted Goal: {}'.format(outputs['goal']))
        for i, beam_output in enumerate(outputs['actions']):
            print("{}: {}".format(i, beam_output))
