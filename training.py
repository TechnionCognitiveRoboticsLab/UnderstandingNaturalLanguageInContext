import torch
import prediction
import pandas as pd
from tqdm import tqdm
import data_validator as validator
from torch.utils.data import DataLoader
from dataset_preprocess import AlfredDataset
from Utils import utils_model, utils_variables, utils_paths, utils_functions, utils_objects
from transformers import T5Tokenizer, T5ForConditionalGeneration


def train_epoch(epoch, tokenizer, model, loader, optimizer):

    model.train()
    running_loss = 0
    i = 0

    tqdm_obj = tqdm(loader)

    for i, data in enumerate(tqdm_obj):

        tqdm_obj.set_description("Batch {}/{}".format(i, len(loader)))

        y = data["target_ids"].to(utils_model.device, dtype=torch.long)
        lm_labels = y[:, :].clone().detach()
        lm_labels[y[:, :] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(utils_model.device, dtype=torch.long)
        mask = data["source_mask"].to(utils_model.device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=lm_labels,
        )

        loss = outputs[0]
        running_loss += loss.item()

        if i > 0:
            tqdm_obj.set_postfix(loss=running_loss/i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('AVG loss for epoch {}: {:.3f}'.format(epoch, running_loss/i))
    return model


def get_loader_from_df(mode,
                       df,
                       tokenizer,
                       model_params,
                       task_type):

    batch_size = model_params[mode.upper() + '_BATCH_SIZE']
    shuffle = False if mode.lower() == 'valid' else True
    num_workers = model_params['num_workers']

    source_text = "input_text"
    target_text = "target_text_{}".format(task_type)

    if target_text not in df.columns:
        target_text = "target_text_both"

    dataset = AlfredDataset(
        df,
        tokenizer,
        model_params["max_src_len"],
        model_params["max_trg_len"],
        source_text,
        target_text,
        task_type
    )

    params = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
    }

    loader = DataLoader(dataset, **params)
    return loader


def train_model(datasets, model_params, task_type, train_size, input_type='task', load_pretrained_model=False):
    dir_path = '{}/{}/{}/{}/{}/'.format(utils_paths.trained_model_path,
                                        model_params["MODEL"],
                                        task_type.capitalize(),
                                        input_type,
                                        train_size)
    utils_paths.check_and_create_dir(dir_path)

    if load_pretrained_model:
        print('\n--- Loading pretrained model from path: {} ---\n'.format(dir_path))
        model, tokenizer = load_model(dir_path)
    else:
        model, tokenizer = load_model(model_params["MODEL"])

    model.config.max_position_embeddings = 1024
    model = model.to(utils_model.device)

    for mode in datasets:
        print(f"{' '.join(mode.split('_')).capitalize()} Dataset len: {datasets[mode].shape[0]}")

    train_loader = get_loader_from_df('train', datasets['train'], tokenizer, model_params, task_type)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        model = train_epoch(epoch, tokenizer, model, train_loader, optimizer)
        model.save_pretrained(dir_path)
        tokenizer.save_pretrained(dir_path)

    print(f"""Model saved at {dir_path}\n""")
    return model


def get_objects_from_relations(relations_split):
    all_obj = set()

    for relation in relations_split:
        objects = relation.split(' ')[1:]
        for obj in objects:
            all_obj.add(obj)

    return all_obj


def clean_goal_predicates(goal_predicates):
    clean_predicates = []

    goal_predicates = goal_predicates.lower()
    for common_mistake in utils_objects.common_goal_mistakes:
        goal_predicates = goal_predicates.replace(common_mistake, '')

    goal_predicates = utils_functions.check_for_duplicate_predicate(goal_predicates)
    for predicate in goal_predicates:
        predicate = prediction.trim_symbols_from_pddl_goal(predicate)
        if not utils_functions.valid_predicate(predicate):
            print('Predicate not valid - {}\n'.format(predicate))
            return []
        clean_predicates.append('({})'.format(predicate))

    if len(goal_predicates) != len(clean_predicates):
        print('Different size goal predicates - {} and {}'.format(goal_predicates, clean_predicates))

    return clean_predicates


def convert_goal_to_pddl(goal_predicates_cleaned):
    num = 2 if '(two_task)' in goal_predicates_cleaned else 1
    total_predicates_wo_two = len(goal_predicates_cleaned) - num + 1

    not_predicate = []
    exists = set()
    predicate_phrase = []

    for predicate in goal_predicates_cleaned:
        predicate = predicate.replace(')', '')
        predicate = predicate.replace('(', '')
        predicate_split = predicate.split(' ')
        predicate_type = predicate_split[0]

        for i in range(num):
            obj1 = '?{}{}'.format(predicate_split[1], i) if len(predicate_split) > 1 else ''
            obj2 = '?{}0'.format(predicate_split[2]) if len(predicate_split) > 2 else ''

            if predicate_type == 'on':
                if len(predicate_split) == 3:
                    exists.add('{} - {}'.format(obj1, predicate_split[1]))
                    exists.add('{} - {}'.format(obj2, predicate_split[2]))
                    predicate_phrase.append('(on {} {})'.format(obj1, obj2))
                else:
                    print('Bad Predicate: {}'.format(predicate))
                    return ''

            elif predicate_type in ['robot_has_obj', 'sliced', 'toggled', 'cleaned', 'cold', 'hot'] \
                    and len(predicate_split) == 2:
                predicate_phrase.append('({} {})'.format(predicate_type, obj1))
                exists.add('{} - {}'.format(obj1, predicate_split[1]))

            elif predicate != 'two_task':
                print('Bad Predicate: {}'.format(predicate))
                return ''

        if num == 2 and len(predicate_split) > 1:
            obj1 = predicate_split[1]
            not_predicate.append(f'(not (= ?{obj1}0 ?{obj1}1))')

    if len(predicate_phrase) not in [total_predicates_wo_two, 2 * total_predicates_wo_two]:
        print(f'total_predicates_wo_two: {total_predicates_wo_two}')
        print(f'predicate_phrase: {predicate_phrase}\n')

    if len(predicate_phrase) == 0:
        print('Empty predicate phrase. orig goal predicate: {}'.format(goal_predicates_cleaned))
        return ''

    exist_phrase = '(exists ({})'.format(' '.join(exists))
    not_equal_phrase = ' '.join(not_predicate)
    predicate_phrase = ' '.join(predicate_phrase)

    goal_predicates_final = '{} (and {} {}) )'.format(exist_phrase, not_equal_phrase, predicate_phrase)

    return goal_predicates_final


def clean_and_insert_pddl_goal_to_problem(problem, goal_predicates, orig_goal_predicates):
    goal_predicates_cleaned = clean_goal_predicates(goal_predicates)
    orig_goal_predicates_cleaned = clean_goal_predicates(orig_goal_predicates)

    if len(goal_predicates_cleaned) == 0:
        return '', []

    goal_predicates_for_pddl = convert_goal_to_pddl(goal_predicates_cleaned)
    if goal_predicates_for_pddl == '':
        return '', goal_predicates_cleaned

    if len(orig_goal_predicates_cleaned) > len(goal_predicates_cleaned):
        print('Different Predicate length. Current: {}, Orig: {}'.format(goal_predicates_cleaned,
                                                                         orig_goal_predicates_cleaned))

    problem_split = problem.split('\n')

    new_problem = []
    for i in range(len(problem_split)):

        new_problem.append(problem_split[i])

        if 'goal' in problem_split[i]:
            new_problem += [goal_predicates_for_pddl]
            new_problem += [')))']
            break

    return '\n'.join(new_problem), goal_predicates_cleaned


def create_template_from_actions(problem_with_meta_data, action_seq, objects_dict):
    problem_split = problem_with_meta_data.split('\n')
    actions = utils_functions.split_predicate_seq(action_seq)
    steps = [f's{i}' for i in range(len(actions) + 1)]
    steps = ' '.join(steps) + ' - stepnumber'
    next_predicates = [f'(next s{i} s{i+1})' for i in range(len(actions))]

    allow = []
    for j, action in enumerate(actions):
        action_params = prediction.get_all_args_from_action(action)[0]
        allow.append('(allowed_{} s{})'.format(action_params['command'], j))
        for k, (param_name, param_val) in enumerate(action_params.items()):
            if param_val in objects_dict and param_name != 'command':
                for candidate in objects_dict[param_val]:
                    allow.append('(allowed_arg{} s{} {})'.format(k, j, candidate))

    new_problem = []
    for line in problem_split:
        new_problem.append(line)
        if ':objects' in line:
            new_problem.append(steps)
        elif ':init' in line:
            new_problem.append('(current_step s0)')
            new_problem += next_predicates
            new_problem += allow
        elif ':goal' in line:
            new_problem.append(f'(current_step s{len(actions)})')

    return '\n'.join(new_problem)


def calc_pddl_correct(actual, predictions, df, mode, model_type, input_type, goal_col_name, both=''):
    print('\nCalculating PDDL correct for model {}, mode {} and goal columns name - {}'.format(model_type.upper(),
                                                                                               mode.upper(),
                                                                                               goal_col_name.upper()))

    tmp_problem_path = utils_paths.problem_path + 'tmp_problem.pddl'
    problems = df['pddl problem'].tolist()
    objects = df['obj_meta'].tolist()
    goal_predicate_orig = df['orig_goal'].tolist()
    goal_predicate = df[goal_col_name].tolist()

    equal = 0
    correct_total = 0
    not_exist_sol = 0

    true_after_pddl = []

    tqdm_obj = tqdm(enumerate(actual), total=len(actual))

    for i, true_actions in tqdm_obj:
        objects_dict = utils_functions.create_object_type_dict(objects[i])
        true_action_seq = true_actions.replace(' .', '.')

        problem = problems[i]
        pred_list = predictions[i].split('|')

        problem_with_meta_data, goal_predicates_cleaned = \
            clean_and_insert_pddl_goal_to_problem(problem, goal_predicate[i], goal_predicate_orig[i])

        if problem_with_meta_data == '':
            continue

        solution_exist = False
        found_equal = False

        for j, action_seq in enumerate(pred_list):
            if action_seq == '':
                continue
            if action_seq[-1] != '.':
                action_seq += '.'
            if true_action_seq == action_seq:
                equal += 1
                found_equal = True
                break

            problem_template = create_template_from_actions(problem_with_meta_data, action_seq, objects_dict)

            with open(tmp_problem_path, 'w') as d:
                for line in problem_template:
                    d.write(line)

            solution_exist, solution = validator.find_solution(tmp_problem_path)

            if solution_exist:
                correct_total += 1

                true_after_pddl.append([true_action_seq, pred_list[0], action_seq, solution, j,
                                        goal_predicate[i], ', '.join(goal_predicates_cleaned),
                                        goal_predicate_orig[i], problem_template])
                break

        if not solution_exist and not found_equal:
            not_exist_sol += 1

    pddl_result_dir = '{}/{}/{}/{}/Predictions/PDDL/'.format(utils_paths.results_path_csv,
                                                             model_type,
                                                             both + 'Actions',
                                                             input_type)

    utils_paths.check_and_create_dir(pddl_result_dir)

    pd.DataFrame(true_after_pddl, columns=['True Action Sequence ',
                                           'First Prediction Sequence',
                                           'Valid Pred Action Sequence',
                                           'Solution Sequence Found',
                                           'Beam Num',
                                           'Goal before cleaning',
                                           'Goal after cleaning',
                                           'Orig Goal Predicate',
                                           'Problem Template']).\
            to_csv('{}pddl_predictions_{}_{}.csv'.format(pddl_result_dir, mode, goal_col_name.split('_')[0] + '_goal'))

    total = equal + not_exist_sol + correct_total
    print(f'\nTotal: {total}')
    print('Correct Equal: {}/{} = {}'.format(equal, total, round(equal/total, 2)))
    print('Correct PDDL: {}/{} = {}'.format(correct_total, total, round(correct_total/total, 2)))
    print('Not Exist Solution: {}/{} = {}\n\n'.format(not_exist_sol, total, round(not_exist_sol/total, 2)))
    return correct_total + equal


def split_df_to_goal_and_actions(dataset):
    goal_df = dataset[dataset.prefix_both == utils_variables.prefixes['goal']].reset_index()
    actions_df = dataset[dataset.prefix_both == utils_variables.prefixes['actions']].reset_index()
    return goal_df, actions_df


def eval_all(datasets, model_params, task_type, input_type, model_type, beam, server, both, train_size):
    both_add = 'Both_' if both else ''
    model_task_type = 'Both' if both else task_type.capitalize()
    dir_path = '{}/{}/{}/{}/{}'.format(utils_paths.trained_model_path, model_params["MODEL"],
                                       model_task_type, input_type, train_size)
    model, tokenizer = load_model(dir_path)

    precision = {}
    recall = {}
    precision_per_value = {}
    recall_per_value = {}

    modes_for_eval = ['val', 'test', 'test_unseen']
    # modes_for_eval = ['test_unseen']
    for mode in modes_for_eval:
        loader = get_loader_from_df('valid', datasets[mode], tokenizer, model_params, task_type)

        predictions_first, predictions, actual = prediction.validate(tokenizer, model, loader, model_params, beam)

        datasets[mode]['orig_{}'.format(task_type)] = actual
        datasets[mode]['pred_{}_beam'.format(task_type)] = predictions

        beam_data_dir = '{}{}/{}/{}/{}/{}/'.format(utils_paths.results_path_csv,
                                                   model_type,
                                                   both_add + task_type.capitalize(),
                                                   input_type,
                                                   train_size,
                                                   'Beam_Data')

        utils_paths.check_and_create_dir(beam_data_dir)

        pd.DataFrame(datasets[mode]).to_csv('{}{}_data_with_beam_predictions.csv'.format(beam_data_dir, mode))

        if not server and task_type == 'actions':
            correct_after_pddl_validation = calc_pddl_correct(actual,
                                                              predictions,
                                                              datasets[mode],
                                                              mode,
                                                              model_type,
                                                              input_type,
                                                              'target_text_goal')
        else:
            correct_after_pddl_validation = 0

        precision_dict, precision_p_value, recall_dict, recall_p_value = \
            prediction.acc_and_print(actual,
                                     predictions_first,
                                     task_type,
                                     mode,
                                     model_params['INPUT'],
                                     model_params['MODEL'],
                                     correct_after_pddl_validation,
                                     both_add,
                                     train_size)

        precision[mode] = precision_dict
        recall[mode] = recall_dict
        precision_per_value[mode] = precision_p_value
        recall_per_value[mode] = recall_p_value

    return {'Precision_General': precision,
            'Precision_per_value': precision_per_value,
            'Recall_General': recall,
            'Recall_per_value': recall_per_value}


def load_model(path):
    model = T5ForConditionalGeneration.from_pretrained(path).to(utils_model.device)
    tokenizer = T5Tokenizer.from_pretrained(path)
    return model, tokenizer


def get_data_dict(task_type, datasets):
    data_dict = {task_type: datasets}
    if task_type == 'both':
        goal_datasets = {}
        actions_datasets = {}
        for mode in datasets:
            goal_data, actions_data = split_df_to_goal_and_actions(datasets[mode])
            goal_datasets[mode] = goal_data
            actions_datasets[mode] = actions_data
        data_dict = {'goal': goal_datasets, 'actions': actions_datasets}
    return data_dict
