import os
import json
import scene
import random
import subprocess
import itertools
from Utils import utils_obj_names
import pandas as pd
from tqdm import tqdm
import problem_generator
from copy import deepcopy
from collections import defaultdict
from Utils import utils_variables, utils_objects, utils_functions, utils_paths


def command_available(command):
    try:
        subprocess.check_call(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, OSError) as err:
        return False


def validator_available():
    return command_available(["Validate", "-h"])


def find_solution(problem_path):
    cmd = ["ff", '-o', utils_paths.domain_path + 'with_template/ALFRED_domain.pddl', '-f', problem_path]
    test = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = test.communicate()[0]
    output = str(output.lower()).split('\\n')
    solution = []
    for i, row in enumerate(output):
        if 'found legal plan as follows' in row:
            for j in range(i + 1, len(output)):
                new_row = output[j]
                if '{}:'.format(len(solution)) in new_row and 'reach-goal' not in new_row:
                    solution.append(new_row.split('{}: '.format(len(solution)))[-1])
            return True, '.'.join(solution)
    return False, '.'.join(solution)


def validate_with_output_reading(pddl_actions, problem_path):
    if not validator_available():
        print('Validate command was not found in this PC. please install KCL plan validator and try again')
        return False

    tmp_path = utils_paths.data_path + 'tmp.soln'
    write_solution(path=tmp_path, actions=pddl_actions)
    cmd = ["Validate", '-v', utils_paths.domain_path + 'without_template/ALFRED_domain.pddl', problem_path, tmp_path]
    test = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = test.communicate()[0]
    output = str(output.lower()).split('\\n')
    return 'plan valid' in output


def add_goto_if_needed(actions_for_pddl, actions, pddl_receptacle, new_names):
    if '(gotolocation {})'.format(pddl_receptacle) not in actions_for_pddl:
        actions_for_pddl.append('(gotolocation {})'.format(pddl_receptacle))
        actions.append('go to {}'.format(new_names[pddl_receptacle]))
        utils_variables.goto_action_added += 1

    return actions_for_pddl, actions


def get_pddl_obj(disc_action, object_target, planner_action_dict):
    utils_variables.missing_main_obj[disc_action] += 1
    if disc_action in ['heat', 'cool', 'put']:
        return object_target
    elif disc_action == 'clean':
        return planner_action_dict['coordinateObjectId'][0].lower()


def get_object_for_receptacle_action(actions_for_pddl,
                                     actions,
                                     disc_action,
                                     pddl_obj,
                                     planner_action_dict,
                                     new_names):

    exist_receptacle = True
    pddl_receptacle = ''

    if 'coordinateReceptacleObjectId' in planner_action_dict:
        pddl_receptacle = planner_action_dict['coordinateReceptacleObjectId'][0].lower()

    else:
        exist_receptacle = False
        utils_variables.missing_receptacle_obj[disc_action] += 1
        if disc_action == 'pick up':
            utils_variables.pickup_only_one_actions += 1
            actions_for_pddl.append('(pickupobject_only_one {})'.format(pddl_obj))
            actions.append('pick up {}'.format(new_names[pddl_obj]))

    return actions_for_pddl, actions, pddl_receptacle, exist_receptacle


def check_pddl_obj_and_planner_obj(disc_action_dict, planner_action_dict):
    pddl_obj = disc_action_dict['args'][0].lower()

    if 'coordinateObjectId' in planner_action_dict:
        pddl_planner_obj = planner_action_dict['coordinateObjectId'][0].lower()
        if pddl_obj != pddl_planner_obj:
            pddl_obj = pddl_planner_obj

    return pddl_obj


def get_actions_from_ann(js):
    actions_for_pddl = []
    actions = []
    slicing_obj_in_scene = []

    high_pddl = js['plan']['high_pddl']

    new_names = pd.read_csv(utils_paths.csv_path + 'new_object_name.csv')
    new_names = dict(zip(new_names.orig_name, new_names.new_name))

    pddl_receptacle = ''
    object_target = js['pddl_params']['object_target'].lower()

    for action_box in high_pddl:
        disc_action_dict = action_box['discrete_action']
        planner_action_dict = action_box['planner_action']

        disc_action_pddl = disc_action_dict['action'].lower()
        disc_action = utils_objects.actions_dict[disc_action_pddl]

        planner_action_pddl = planner_action_dict['action'].lower()
        planner_action = utils_objects.actions_dict[planner_action_pddl]

        if disc_action in ['noop', 'end'] or planner_action in ['noop', 'end']:
            continue
        else:
            pddl_obj = check_pddl_obj_and_planner_obj(disc_action_dict, planner_action_dict)

        if disc_action in ['go to', 'toggle']:
            if disc_action == 'go to' and pddl_obj not in utils_objects.receptacles:
                utils_variables.direct_object_command += 1

            actions.append("{} {}".format(disc_action, new_names[pddl_obj]))
            actions_for_pddl.append("({} {})".format(disc_action_pddl, pddl_obj))
            continue

        elif disc_action in ['pick up', 'heat', 'cool', 'put', 'clean']:
            if pddl_obj == '':
                pddl_obj = get_pddl_obj(disc_action, object_target, planner_action_dict)

            actions_for_pddl, actions, pddl_receptacle, exist_recept = \
                get_object_for_receptacle_action(actions_for_pddl, actions, disc_action,
                                                  pddl_obj, planner_action_dict, new_names)

            if disc_action == 'pick up':
                slicing_obj_in_scene.append(pddl_obj)
                if not exist_recept:
                    continue

        elif disc_action == 'slice':
            pddl_receptacle = slicing_obj_in_scene[-1]
            if 'knife' not in pddl_receptacle.lower():
                utils_variables.missing_receptacle_obj[disc_action] += 1
            # actions_for_pddl, actions = add_goto_if_needed(actions_for_pddl, actions, pddl_obj, new_names)

        if pddl_obj == '':
            print('Empty pddl name!')

        actions.append("{} {} {}".format(planner_action, new_names[pddl_obj], new_names[pddl_receptacle]))
        actions_for_pddl.append("({} {} {})".format(planner_action_pddl, pddl_obj, pddl_receptacle))

    return ' . '.join(actions) + ' .', actions_for_pddl


def find_obj_id_in_plan(plan, predicate_type, two_flag, obj, action_name='', only_obj=False):
    found = False
    possible_goals = []

    for action in plan:
        action_name_from_plan = utils_functions.get_id_from_action(action, 'action')
        if action_name:
            if action_name_from_plan != action_name:
                continue

        if predicate_type == 'cleaned':
            obj_id = utils_functions.get_id_from_action(action, 'cleanObjectId')
        elif 'objectId' in action['planner_action']:
            obj_id = utils_functions.get_id_from_action(action)
        else:
            continue
        obj_short_name = obj_id.split('|')[0]

        if obj_short_name == obj:
            if not only_obj:
                predicate = '{} {}'.format(predicate_type, obj_id)
            else:
                predicate = obj_id
            if predicate not in possible_goals:
                if found and not two_flag:
                    print('Found two elements to be {}!'.format(predicate_type))
                possible_goals.append(predicate)
                found = True

    return possible_goals


def find_specific_obj_id_in_plan(obj, plan):
    found = False
    possible_ids = []

    for action in plan:
        if 'objectId' in action["planner_action"]:
            obj_id = utils_functions.get_id_from_action(action)
            obj_short_name = obj_id.split('|')[0]

            if obj_short_name == obj:
                if obj_id not in possible_ids:
                    if found:
                        print('Found two elements of obj {} in plan!'.format(obj_short_name))
                    found = True
                    possible_ids.append(obj_id)
    return possible_ids


def find_obj_id_and_recep_id(obj, recep, plan, two_flag):
    found = False
    possible_goals = []

    for action in plan:
        action_name = utils_functions.get_id_from_action(action, 'action')

        if action_name == 'putobject':
            obj_full_name = utils_functions.get_id_from_action(action)
            obj_short_name = obj_full_name.split('|')[0]

            recept_full_name = utils_functions.get_id_from_action(action, 'receptacleObjectId')
            recept_short_name = recept_full_name.split('|')[0]

            if obj_short_name == obj and recept_short_name == recep:
                predicate = utils_functions.in_or_on(recept_full_name, obj_full_name)
                if predicate not in possible_goals:
                    if found and not two_flag:
                        possible_goals = [predicate]
                    else:
                        possible_goals.append(predicate)
                    found = True

    return possible_goals


def find_relations(relations, obj, recept):
    for relation in relations:
        relation = relation.lower()
        if obj in relation and recept in relation:
            return relation


def check_for_exist_in_relations(receptacle_target, parent_target, plan, relations_with_ids, two_flag):
    receptacle_target_id = find_obj_id_in_plan(plan, predicate_type='', two_flag=two_flag, obj=receptacle_target,
                                               action_name='', only_obj=True)
    parent_target_id = find_obj_id_in_plan(plan, predicate_type='', two_flag=two_flag, obj=parent_target,
                                               action_name='', only_obj=True)

    if len(receptacle_target_id) == 0 and len(parent_target_id) > 0:
        for parent_id in parent_target_id:
            relation = find_relations(relations_with_ids,
                                                        parent_id.lower(),
                                                        receptacle_target)
            if relation:
                return [relation]

    elif len(parent_target_id) == 0 and len(receptacle_target_id) > 0:
        for recept_id in receptacle_target_id:
            relation = find_relations(relations_with_ids,
                                                    recept_id.lower(),
                                                    parent_target)
            if relation:
                return [relation]

    else:
        relation = find_relations(relations_with_ids, receptacle_target, parent_target)
        if relation:
            return [relation]


def check_for_exist_in_objects(obj, objects):
    possible_matches = set()
    for potential_object in objects:
        if potential_object.split('|')[0].lower() == obj:
            possible_matches.add(potential_object)
    return possible_matches


def validate_predicate(pddl_predicates):

    # Check for same object in different location
    on_predicates = [x for x in pddl_predicates if 'on ' in x]
    hot_predicates = [x for x in pddl_predicates if 'hot ' in x]
    cold_predicates = [x for x in pddl_predicates if 'cold ' in x]

    for predicate in on_predicates:
        split = predicate.split(' ')
        if 'on' == split[0]:
            obj = split[1]
            for predicate2 in on_predicates:
                if predicate2 != predicate and 'on {}'.format(obj) in predicate2:
                    return False

    # Check for same object to be hot and cold
    for hot_predicate in hot_predicates:
        for cold_predicate in cold_predicates:
            if hot_predicate.split(' ')[1] == cold_predicate.split(' ')[1]:
                return False

    return True


def convert_sample_to_pddl_goal_with_ids(sample, pddl_goal, relations_with_ids, objects):
    pddl_params = sample['pddl_params']
    plan = sample['plan']['high_pddl']

    obj = pddl_params['object_target'].lower()
    receptacle_target = pddl_params['mrecep_target'].lower()
    parent_target = pddl_params['parent_target'].lower()

    pddl_goal_with_ids = []

    two_flag = True if 'two' in sample['task_type'] else False

    for goal_predicate in pddl_goal:
        if 'two_task' in goal_predicate:
            continue
        split_predicate = goal_predicate.lower().split(' ')
        main_predicate_obj = split_predicate[1]
        if len(split_predicate) == 2:
            if 'sliced' in goal_predicate:
                goal_with_id = find_obj_id_in_plan(plan, 'sliced', two_flag, main_predicate_obj, 'sliceobject')
                pddl_goal_with_ids += goal_with_id

            elif 'toggled' in goal_predicate:
                goal_with_id = find_obj_id_in_plan(plan, 'toggled', two_flag, main_predicate_obj, 'toggleobject')
                goal_with_id_has_obj = find_obj_id_in_plan(plan, 'robot_has_obj', two_flag, obj)
                pddl_goal_with_ids += goal_with_id
                pddl_goal_with_ids += goal_with_id_has_obj

            elif 'hot' in goal_predicate:
                goal_with_id = find_obj_id_in_plan(plan, 'hot', two_flag, obj)
                pddl_goal_with_ids += goal_with_id

            elif 'cold' in goal_predicate:
                goal_with_id = find_obj_id_in_plan(plan, 'cold', two_flag, obj)
                pddl_goal_with_ids += goal_with_id

            elif 'clean' in goal_predicate:
                goal_with_id = find_obj_id_in_plan(plan, 'cleaned', two_flag, main_predicate_obj, 'cleanobject')
                pddl_goal_with_ids += goal_with_id

    if parent_target and not receptacle_target:
        goal_with_id = find_obj_id_and_recep_id(obj, parent_target, plan, two_flag)
        if len(goal_with_id) == 0:
            goal_with_id = check_for_exist_in_relations(obj, parent_target, plan, relations_with_ids, two_flag)
        pddl_goal_with_ids += goal_with_id

    if parent_target and receptacle_target:
        goal_with_id = find_obj_id_and_recep_id(receptacle_target, parent_target, plan, two_flag)
        if len(goal_with_id) == 0:
            goal_with_id = check_for_exist_in_relations(receptacle_target, parent_target, plan, relations_with_ids,
                                                        two_flag)
        pddl_goal_with_ids += goal_with_id

        goal_with_id = find_obj_id_and_recep_id(obj, receptacle_target, plan, two_flag)
        if not goal_with_id:
            goal_with_id = check_for_exist_in_relations(obj, receptacle_target, plan, relations_with_ids, two_flag)
        if not goal_with_id:
            obj_id_list = check_for_exist_in_objects(obj, objects)
            recep_id_list = check_for_exist_in_objects(receptacle_target, objects)
            if len(obj_id_list) == 1 and len(recep_id_list) == 1:
                goal_with_id = ['on {} {}'.format(obj_id_list, recep_id_list)]
            if not goal_with_id:
                utils_variables.bad_goal += len(sample['turk_annotations']['anns'])
                return []
        pddl_goal_with_ids += goal_with_id

    if receptacle_target and not parent_target:
        goal_with_id = find_obj_id_and_recep_id(obj, receptacle_target, plan, two_flag)
        if len(goal_with_id) == 0:
            goal_with_id = check_for_exist_in_relations(obj, receptacle_target, plan, relations_with_ids, two_flag)
        pddl_goal_with_ids += goal_with_id

    valid_predicate = validate_predicate(pddl_goal_with_ids)
    if valid_predicate:
        return pddl_goal_with_ids
    return []


def convert_sample_to_pddl_goal(sample, sample_type, two_task):
    pddl_goal = []

    pddl_params = sample['pddl_params']
    obj = pddl_params['object_target'].lower()
    sliced = pddl_params['object_sliced']
    receptacle_target = pddl_params['mrecep_target'].lower()
    parent_target = pddl_params['parent_target'].lower()
    toggled = pddl_params['toggle_target'].lower()

    if sliced:
        pddl_goal.append('sliced {}'.format(obj))

    if toggled:
        pddl_goal.append('toggled {}'.format(toggled))
        pddl_goal.append('robot_has_obj {}'.format(obj))

    if parent_target and not receptacle_target:
        pddl_goal.append(utils_functions.in_or_on(parent_target, obj))

    if parent_target and receptacle_target:
        pddl_goal.append(utils_functions.in_or_on(receptacle_target, obj))
        pddl_goal.append(utils_functions.in_or_on(parent_target, receptacle_target))

    if receptacle_target and not parent_target:
        pddl_goal.append(utils_functions.in_or_on(receptacle_target, obj))

    if 'heat' in sample_type:
        pddl_goal.append('hot {}'.format(obj))

    if 'cool' in sample_type:
        pddl_goal.append('cold {}'.format(obj))

    if 'clean' in sample_type:
        pddl_goal.append('cleaned {}'.format(obj))

    if two_task:
        pddl_goal.append('two_task')

    return ['({})'.format(x) for x in pddl_goal], pddl_goal


def check_plan_structure(plan):
    new_plan = []
    for sentence in plan:
        new_sentence = [sentence.lower()]
        new_sentence = utils_functions.trim_spaces(new_sentence)[0]
        if new_sentence[-1] != '.':
            new_sentence += '.'
        new_plan.append(new_sentence)

    new_plan = ''.join(new_plan)
    new_plan = '. '.join(new_plan.split('.'))[:-1]
    return new_plan


def write_solution(path, actions):
    if os.path.isfile(path):
        os.remove(path)
    with open(path, 'w') as f:
        for action in actions:
            f.write("{}\n".format(action))


def change_goal_to_lang(row):
    new_goal = row.lower()
    for symbol in [')', '(']:
        new_goal = new_goal.replace(symbol, '')

    # for short_obj, long_obj in utils_obj_names.obj_short_to_long.items():
    #     new_goal = new_goal.replace(short_obj, long_obj)

    for short_goal, long_goal in utils_obj_names.goal_long_to_short.items():
        new_goal = new_goal.replace(short_goal, long_goal)

    return new_goal


def insert_data_into_dict(big_data_dict,
                          task_id,
                          ann_val,
                          samples_counter,
                          problem_path,
                          sol_path,
                          actions,
                          goal_predicates,
                          valid_add,
                          obj_list,
                          objects_relations,
                          random_relations_dict=None,
                          meta_data=None):

    sample_dict = {'task': '',
                   'plan': '',
                   'goal': '',
                   'actions': '',
                   'pddl_actions': '',
                   'pddl problem': '',
                   'solution valid': ''}

    for j in range(len(ann_val)):
        random.shuffle(obj_list)
        random.shuffle(list(objects_relations))

        big_data_dict[samples_counter] = deepcopy(sample_dict)
        big_data_dict[samples_counter]['task_id'] = task_id
        big_data_dict[samples_counter]['task'] = utils_functions.clean_val_for_preprocess(ann_val[j]['task_desc'])
        big_data_dict[samples_counter]['plan'] = check_plan_structure(ann_val[j]['high_descs'])
        big_data_dict[samples_counter]['goal'] = goal_predicates.lower()
        big_data_dict[samples_counter]['actions'] = actions.lower()
        big_data_dict[samples_counter]['objects'] = ', '.join(obj_list).lower()

        if len(list(objects_relations)) == 0:
            big_data_dict[samples_counter]['relations_meta'] = ''
        else:
            big_data_dict[samples_counter]['relations_meta'] = ', '.join(list(objects_relations))

        if random_relations_dict:
            for noise, noised_relation in random_relations_dict.items():
                if len(list(noised_relation)) == 0:
                    big_data_dict[samples_counter]['NoisedRelations'+str(noise)] = ''
                    continue
                random.shuffle(noised_relation)
                big_data_dict[samples_counter]['NoisedRelations'+str(noise)] = ' , '.join(list(noised_relation))

        # big_data_dict[samples_counter]['goal_in_lang'] = change_goal_to_lang(big_data_dict[samples_counter]['goal'])

        with open(sol_path, 'r') as file:
            big_data_dict[samples_counter]['pddl_actions'] = file.read()

        with open(problem_path, 'r') as file:
            big_data_dict[samples_counter]['pddl problem'] = file.read()

        big_data_dict[samples_counter]['solution valid'] = valid_add

        for key in ['relations_meta_wo_ids', 'obj_meta']:
            big_data_dict[samples_counter][key] = ', '.join(meta_data[key])
        samples_counter += 1

    return big_data_dict, samples_counter


def print_statistics(valid_plans, non_valid_plans, bad_problems, mode):
    print("Mode: {}".format(mode))
    total = valid_plans + non_valid_plans + bad_problems
    total_wo_bad = valid_plans + non_valid_plans
    print("Total problems: {}\n"
          "Total bad problems: {}\n"
          "Total valid plans: {}/{} = {:.2f}\n"
          "Total non-valid plans: {}/{} = {:.2f} ".format(total,
                                                          bad_problems,
                                                          valid_plans,
                                                          total_wo_bad,
                                                          valid_plans / total_wo_bad,
                                                          non_valid_plans,
                                                          total_wo_bad,
                                                          non_valid_plans / total_wo_bad))

    print("Direct object commands: {}".format(utils_variables.direct_object_command))
    print("Total GoTo actions added: {}".format(utils_variables.goto_action_added))
    print("Total Missing main obj per action: {}".format(utils_variables.missing_main_obj))
    print("Total Missing receptacle obj per action: {}".format(utils_variables.missing_receptacle_obj))
    print("Total Pickup only one actions: {}".format(utils_variables.pickup_only_one_actions))


def check_if_solution_valid(pddl_actions, problem_path, valid_plans=0, non_valid_plans=0):
    z = validate_with_output_reading(pddl_actions, problem_path)

    if z:
        valid_add = 'Valid'
        valid_plans += 1
    else:
        valid_add = 'Non valid'
        non_valid_plans += 1

    return valid_add, valid_plans, non_valid_plans


def save_json_and_csv(big_data_dict, mode):
    mode = mode.replace('/', '_').lower()
    path_with_mode_csv = utils_paths.csv_path + mode
    path_with_mode_json = utils_paths.jsons_path + mode

    with open(path_with_mode_json + '_data_with_pddl.json', 'w') as d:
        json.dump(big_data_dict, d)

    pd.DataFrame(big_data_dict).T.to_csv(path_with_mode_csv + '_data_with_pddl.csv', index=False)


def match_lists(list_a, list_b):
    result_list = []
    len_b = len(list_b)
    for obj in list_a:
        i = random.randint(0, len_b - 1)
        result_list.append('on {} {}'.format(obj, list_b[i]))
    return result_list


def match_and_cut(list_a, list_b, p):
    if len(list_a) == 0 or len(list_b) == 0:
        return [], list_a
    matched = match_lists(list_a, list_b)
    random.shuffle(matched)
    k = int(p * len(list_a))
    matched = matched[:k]
    non_matched = [obj for obj in list_a if obj not in ' '.join(matched)]
    return matched, non_matched


def create_random_relations(objects, noise=0.5):
    mobile = [obj for obj in objects if obj not in utils_objects.receptacles]
    receptacles = [obj for obj in objects if obj in utils_objects.receptacles]
    receptacle_mobile = [obj for obj in receptacles if obj in utils_objects.receptacles_mobile]
    receptacle_non_mobile = [obj for obj in receptacles if obj in utils_objects.receptacles_non_mobile]

    """
    m = mobile
    rm = receptacle_mobile
    rnm = receptacle_non_mobile
    """
    m_rm_p = 0.25
    rm_rnm_p = 0.25

    # Mobile On Mobile
    m_on_rm, m_left = match_and_cut(mobile, receptacle_mobile, m_rm_p)

    # Mobile Receptacle On Non-Mobile Receptacle
    rm_on_rnm, rm_left = match_and_cut(receptacle_mobile, receptacle_non_mobile, rm_rnm_p)

    # Mobile On Non-Mobile Receptacle
    m_on_rnm, m_left = match_and_cut(m_left, receptacle_non_mobile, p=1)

    new_relations = m_on_rm + rm_on_rnm + m_on_rnm
    random.shuffle(new_relations)
    return new_relations[:int(len(new_relations)*noise)]


def convert_goal_id_to_names(id_to_name_dict, goal_with_ids):
    new_goal = []
    for goal_predicate in goal_with_ids:
        split = goal_predicate.split(' ')
        goal_type = split[0]
        obj1 = id_to_name_dict[split[1]]
        if len(split) == 3:
            obj2 = id_to_name_dict[split[2]]
            new_goal.append('({} {} {})'.format(goal_type, obj1, obj2))
        else:
            new_goal.append('({} {})'.format(goal_type, obj1))

    return new_goal


def convert_id_to_name(obj_id, id_to_name):
    obj_id = utils_functions.extract_obj_id(obj_id)
    return id_to_name[obj_id]


def get_disc_and_planner_info(action_box):
    disc_action_pddl = action_box['discrete_action']['action'].lower()
    disc_action = utils_objects.actions_dict[disc_action_pddl]
    planner_action_pddl = action_box['planner_action']['action'].lower()
    
    return disc_action, planner_action_pddl


def create_action_from_objects(action_type, params, pddl=False):
    start = finish = ''
    if pddl:
        start = '('
        finish = ')'
    
    action = '{}{}'.format(start, action_type)
    for arg in params:
        action += ' {}'.format(arg)
    action += finish
    
    return action


def get_obj_num_name(action_box, id_to_name, key='objectId'):
    obj_id = utils_functions.get_id_from_action(action_box, key)
    obj_name = id_to_name[obj_id]
    return obj_name


def reg_name_from_action_box(action_box, id_to_name, key='objectId'):
    if key == 'coordinateReceptacleObjectId':
        obj_name = action_box["planner_action"][key][0].lower()
    else:
        obj_name = get_obj_num_name(action_box, id_to_name, key=key)
    return obj_name.split('_')[0]


def reg_name_from_id(obj_id, id_to_name):
    obj_name = id_to_name[obj_id]
    return obj_name.split('_')[0]


def get_objects_and_params(obj_name, recept_name=''):
    if recept_name:
        params_names = [obj_name, recept_name]
    else:
        params_names = [obj_name]

    params_wo_ids = params_names
    return params_names, params_wo_ids


def insert_action(disc_action,
                  planner_action_pddl,
                  params_names, 
                  params_wo_ids, 
                  actions, 
                  actions_pddl):
    
    actions.append(create_action_from_objects(disc_action, params_wo_ids))
    actions_pddl.append(create_action_from_objects(planner_action_pddl, params_names, True))
    
    return actions, actions_pddl


def find_last_picked_element(obj_only_name, elements_picked_up):
    for k in range(len(elements_picked_up) - 1, -1, -1):
        if obj_only_name == elements_picked_up[k].split('|')[0]:
            return elements_picked_up[k]
    return ''


def find_recept_id_in_actions(name_to_id, actions_for_pddl, recept_name):
    for pddl_action in actions_for_pddl:
        if recept_name in pddl_action and 'goto' in pddl_action:
            return name_to_id[pddl_action.split(' ')[-1].replace(')', '')]
    return ''


def find_recept_id_for_pickup(obj_name, action_box, name_to_id, id_to_name, all_objects, relations, actions_for_pddl):
    recept_name = reg_name_from_action_box(action_box, id_to_name, 'coordinateReceptacleObjectId')
    all_instances = [x for x in all_objects if x.split('|')[0].lower() == recept_name]
    if len(all_instances) == 1:
        return utils_functions.extract_obj_id(all_instances[0])
    else:
        for i in range(len(actions_for_pddl) - 1, -1, -1):
            pddl_action = actions_for_pddl[i]
            pddl_action_split = pddl_action.split(' ')
            if obj_name == pddl_action_split[1] and 'putobject' in pddl_action_split[0]:
                return name_to_id[pddl_action.split(' ')[-1].replace(')', '')]

    recept_id = find_recept_id_in_relations(name_to_id, relations, obj_name)
    if recept_id:
        return recept_id
    else:
        return find_recept_id_in_actions(name_to_id, actions_for_pddl, recept_name)


def find_recept_id_in_relations(name_to_id, relations, obj_name, recept_reg_name=''):
    for relation in relations:
        relation_split = relation.split(' ')
        if recept_reg_name:
            if obj_name == relation_split[1] and len(relation_split) == 3 and recept_reg_name in relation:
                return name_to_id[relation_split[-1]]
        else:
            if obj_name == relation_split[1] and len(relation_split) == 3:
                return name_to_id[relation_split[-1]]


def is_same_obj(action_box, id_to_name, obj_reg_name, name_to_id, key='objectId'):
    put_obj_id = name_to_id[get_obj_num_name(action_box, id_to_name, key=key)]
    if put_obj_id.split('|')[0] == obj_reg_name:
        return put_obj_id
    return ''


def add_relation_if_needed(relations, obj_name, recept_name, actions_for_pddl):
    exist_in_relation = False
    exist_in_actions = False

    for action in actions_for_pddl:
        if 'pickupobject {}'.format(obj_name) in action:
            exist_in_actions = True

    for relation in relations:
        if 'on {}'.format(obj_name) in relation:
            exist_in_relation = True

    if not exist_in_actions and not exist_in_relation:
        if recept_name and obj_name:
            relations.append('on {} {}'.format(obj_name, recept_name))
            utils_variables.added_relation += 1

    return relations


def goto_check(actions, pddl_actions):
    new_actions = []
    new_pddl_actions = []
    actions_split = actions.split(' . ')

    new_pddl_actions.append(pddl_actions[0])
    new_actions.append(actions_split[0])

    for i in range(1, len(pddl_actions)):
        current_action_split = pddl_actions[i].split(' ')
        prev_action_split = pddl_actions[i - 1].split(' ')

        if current_action_split[0] != '(gotolocation' and prev_action_split[0] != '(gotolocation':
            current_recept = ''
            prev_recept = ''
            if len(current_action_split) == 3:
                current_recept = current_action_split[2]
            if len(prev_action_split) == 3:
                prev_recept = prev_action_split[2]

            if current_recept != prev_recept:
                new_pddl_actions.append('(gotolocation {}'.format(current_recept))
                new_actions.append('go to {}'.format(current_recept.split('_')[0]))

        new_pddl_actions.append(' '.join(current_action_split))
        new_actions.append(actions_split[i])

    return new_actions, new_pddl_actions


def get_actions_from_ann_with_ids(js,
                                  id_to_name, 
                                  name_to_id,
                                  all_objects, 
                                  relations):

    utils_variables.total_anns += len(js['turk_annotations']['anns'])

    actions_for_pddl = []
    actions = []
    elements_picked_up = knives_picked_up = []

    high_pddl = js['plan']['high_pddl']

    bad_ann = False
    utils_variables.relation_added = False

    kb_dict = {key.split(' ')[1]: key.split(' ')[2] for key in relations}

    for i, action_box in enumerate(high_pddl):
        disc_action, pddl_action = get_disc_and_planner_info(action_box)
        
        if disc_action in ['noop', 'end'] or pddl_action in ['noop', 'end']:
            continue
        
        elif disc_action in ['toggle', 'put', 'clean', 'slice']:
            recept_name = obj_name = ''

            if disc_action == 'toggle':
                obj_name = get_obj_num_name(action_box, id_to_name)

            elif disc_action == 'put':
                obj_name = get_obj_num_name(action_box, id_to_name)
                recept_name = get_obj_num_name(action_box, id_to_name, 'receptacleObjectId')
                kb_dict[obj_name] = recept_name

            elif disc_action == 'clean':
                obj_name = get_obj_num_name(action_box, id_to_name, 'cleanObjectId')
                recept_name = get_obj_num_name(action_box, id_to_name)

            elif disc_action == 'slice':
                obj_name = get_obj_num_name(action_box, id_to_name)
                recept_name = id_to_name[knives_picked_up[-1]]

            params_names, params_wo_ids = get_objects_and_params(obj_name, recept_name)
            actions, actions_for_pddl = insert_action(disc_action, pddl_action, params_names, params_wo_ids, actions,
                                                      actions_for_pddl)

        elif disc_action in ['heat', 'cool']:
            recept_name = get_obj_num_name(action_box, id_to_name)

            obj_reg_name = action_box['discrete_action']['args'][0]
            obj_id = find_last_picked_element(obj_reg_name, elements_picked_up)

            # If Args = '' take the last object picked
            if not obj_id:
                obj_id = elements_picked_up[-1]
                utils_variables.no_arg_in_heat_cool += len(js['turk_annotations']['anns'])

            obj_name = id_to_name[obj_id]
            params_names, params_wo_ids = get_objects_and_params(obj_name, recept_name)
            actions, actions_for_pddl = insert_action(disc_action, pddl_action,
                                                      params_names, params_wo_ids,
                                                      actions, actions_for_pddl)

        elif disc_action == 'pick up':
            recept_name = ''
            obj_name = get_obj_num_name(action_box, id_to_name)

            if 'coordinateReceptacleObjectId' in action_box['planner_action']:
                recept_id = find_recept_id_for_pickup(obj_name, action_box, name_to_id, id_to_name,
                                                      all_objects, relations, actions_for_pddl)
            else:
                recept_id = find_recept_id_in_relations(name_to_id, relations, obj_name)

            if 'knife' in obj_name:
                knives_picked_up.append(obj_name)
            elements_picked_up.append(name_to_id[obj_name])

            if not recept_id:
                utils_variables.no_recept_in_pickup += len(js['turk_annotations']['anns'])
                pddl_action = 'pickupobject_only_one'
            else:
                recept_name = id_to_name[recept_id]

            relations = add_relation_if_needed(relations, obj_name, recept_name, actions_for_pddl)
            params_names, params_wo_ids = get_objects_and_params(obj_name, recept_name)
            actions, actions_for_pddl = insert_action(disc_action, pddl_action,
                                                      params_names, params_wo_ids,
                                                      actions, actions_for_pddl)
            kb_dict[obj_name] = ''

        elif disc_action == 'go to':

            obj_reg_name = action_box['discrete_action']['args'][0].lower()
            obj_id = ''

            all_instances = [x for x in all_objects if x.split('|')[0] == obj_reg_name]
            if len(all_instances) == 1:
                obj_id = all_instances[0]

            else:
                for j in range(i + 1, len(high_pddl)):
                    action_box_j = high_pddl[j]
                    planner = action_box_j['planner_action']
                    action_type = planner['action'].lower()
                    if action_type == 'gotolocation':
                        break
                    elif 'objectId' in planner:
                        obj_id = is_same_obj(action_box_j, id_to_name, obj_reg_name, name_to_id)
                        if not obj_id and action_type == 'putobject' and 'receptacleObjectId' in planner:
                            obj_id = is_same_obj(action_box_j, id_to_name, obj_reg_name,
                                                 name_to_id, key='receptacleObjectId')

                        if not obj_id and action_type == 'pickupobject':
                            temp_obj_name = get_obj_num_name(action_box_j, id_to_name)
                            if temp_obj_name not in ' '.join(actions_for_pddl):
                                obj_id = find_recept_id_in_relations(name_to_id, relations, temp_obj_name, obj_reg_name)
                            elif kb_dict[temp_obj_name] != '':
                                obj_id = name_to_id[kb_dict[temp_obj_name]]

                        if obj_id:
                            break

            if not obj_id:
                utils_variables.bad += len(js['turk_annotations']['anns'])
                break
            else:
                params_names, params_wo_ids = get_objects_and_params(id_to_name[obj_id])
                actions, actions_for_pddl = insert_action(disc_action, pddl_action,
                                                          params_names, params_wo_ids,
                                                          actions, actions_for_pddl)
    return ' . '.join(actions) + ' .', actions_for_pddl, bad_ann, relations


def validate_orig_data(mode='Train'):
    mode = mode.split('/')
    mode[0] = mode[0].capitalize()
    mode = '/'.join(mode)

    utils_variables.direct_object_command = 0
    utils_variables.goto_action_added = 0
    utils_variables.pickup_only_one_actions = 0
    meta_valid = 0
    meta_total = 0
    utils_variables.missing_receptacle_obj = defaultdict(int)
    utils_variables.missing_main_obj = defaultdict(int)

    print("Mode: {} \n".format(mode))

    utils_functions.empty_directories()
    valid_plans = non_valid_plans = bad_problems = samples_counter = 0
    tqdm_obj = tqdm(utils_functions.retrieve_all_task_jsons(mode).items())
    metadata_dict = {}
    if utils_variables.use_meta:
        metadata_dict = utils_functions.retrieve_all_task_jsons(mode, 'more_info.json')

    big_data_dict = {}

    scenes_types = defaultdict(int)

    for task_desc, js in tqdm_obj:
        floor_type = js[0]['scene']['floor_plan']
        scenes_types[utils_variables.scene_type_dic[floor_type]] += 1

        task_type = task_desc.split('-')
        task_type = task_type[:len(task_type) - 4]
        task_type = '-'.join(task_type)

        for i, ann in enumerate(js):
            two_task = True if 'two' in task_type else False

            goal_predicates, pddl_goal_list = convert_sample_to_pddl_goal(ann, task_type, two_task)
            actions, pddl_actions = get_actions_from_ann(ann)

            if len(actions) == 0:
                bad_problems += 1
                continue

            obj_list, objects_relations = scene.get_scene_data_from_js(ann, actions.split('.')[:-1])
            objects_not_in_relations = [obj for obj in obj_list if obj not in ' '.join(objects_relations)]

            random_relations_dict = {}
            for noise in utils_variables.noises:
                random_relations = create_random_relations(objects_not_in_relations, noise=noise)
                noised_relations = random_relations + list(objects_relations)
                random.shuffle(noised_relations)
                random_relations_dict[noise] = noised_relations

            this_problem_prefix = "{}_{}".format(task_desc, i)

            meta_data = get_all_meta_actions_relations_objects(metadata_dict[task_desc][i],
                                                               ann,
                                                               this_problem_prefix + '_Meta',
                                                               pddl_goal_list)
            meta_total += 1
            if meta_data['valid_meta'] == 'Valid':
                valid_plans += 1
            elif len(meta_data) == 1:
                bad_problems += 1
                continue
            else:
                non_valid_plans += 1

            problem_path = meta_data['problem_path_meta']
            valid_add = meta_data['valid_meta']
            sol_path = meta_data['sol_path']
            objects_relations = meta_data['relations_meta']

            big_data_dict, samples_counter = insert_data_into_dict(big_data_dict,
                                                                   ann['task_id'],
                                                                   ann['turk_annotations']['anns'],
                                                                   samples_counter,
                                                                   problem_path,
                                                                   sol_path,
                                                                   actions,
                                                                   ', '.join(goal_predicates),
                                                                   valid_add,
                                                                   obj_list,
                                                                   objects_relations,
                                                                   random_relations_dict=None,
                                                                   meta_data=meta_data)

    print_statistics(valid_plans, non_valid_plans, bad_problems, mode)
    print('Mode: {}\nMeta plans valid: {}/{}= {}'.format(mode, meta_valid, meta_total, round(meta_valid/meta_total, 2)))
    print('Mode: {}\nMeta plans Non valid: {}/{} = {}'.format(mode, meta_total-meta_valid, meta_total,
                                                                round((meta_total-meta_valid)/meta_total, 2)))

    print("Mode: {}, More Than One Recept in Metadata : {}".format(mode,
                                                                   utils_variables.more_than_one_recept_in_metadata))
    print('Mode: {}, total num of bad task id: {}'.format(mode, len(utils_variables.bad_tsk_id)))
    print('Mode: {} , Different goal len: {}'.format(mode, utils_variables.different_goal_len))
    print('Mode: {} , goto added: {}'.format(mode, utils_variables.goto_action_added))

    print('Floor types distribution: {}'.format(scenes_types))

    utils_variables.different_goal_len = 0
    utils_variables.goto_action_added = 0
    utils_variables.more_than_one_recept_in_metadata = 0

    mode = 'orig_train' if mode.lower() == 'train' else mode
    save_json_and_csv(big_data_dict, mode)


def create_general_goal_predicates(goal_predicates):
    unique_objects_in_goal = set()
    obj_counter = defaultdict(int)
    id_to_name = {}

    for predicate in goal_predicates:
        predicate_split = predicate.split(' ')
        predicate_type = predicate_split[0]
        unique_objects_in_goal.add(predicate_split[1])
        if predicate_type == 'on':
            unique_objects_in_goal.add(predicate_split[2])

    for obj_id in unique_objects_in_goal:
        obj_type = obj_id.split('|')[0]
        id_to_name[obj_id] = f'?{obj_type}{obj_counter[obj_type]}'
        obj_counter[obj_type] += 1

    exist_objects = []
    not_equal_total = []
    for arg_type, count in obj_counter.items():
        obj_type_list = ['?{}{} - {}'.format(arg_type, i, arg_type) for i in range(count)]
        exist_objects += obj_type_list

        pairs_list = itertools.combinations(list(range(count)), r=2)
        not_equal_total += ['(not (= ?{}{} ?{}{}))'.format(arg_type, x[0], arg_type, x[1]) for x in pairs_list]

    orig_predicates_with_names = []
    for predicate in goal_predicates:
        for obj_id, name in id_to_name.items():
            predicate = predicate.replace(obj_id, name)
        orig_predicates_with_names.append('({})'.format(predicate))

    exist_phrase = '(exists ({})'.format(' '.join(exist_objects))
    not_equal_phrase = ' '.join(not_equal_total)
    predicate_phrase = ' '.join(orig_predicates_with_names)

    goal_predicates_final = '{} (and {} {}) )'.format(exist_phrase, not_equal_phrase, predicate_phrase)

    return [goal_predicates_final]


def get_all_meta_actions_relations_objects(metadata,
                                           ann,
                                           this_problem_prefix,
                                           pddl_goal_list):

    obj_meta_ids, obj_meta, relations_meta, relations_meta_ids, id_to_name, name_to_id, relations_reg_names = \
        scene.get_scene_info_from_metadata(metadata)

    actions_with_ids, pddl_actions, bad_ann, relations_meta = get_actions_from_ann_with_ids(ann,
                                                                                            id_to_name,
                                                                                            name_to_id,
                                                                                            obj_meta_ids,
                                                                                            relations_meta)

    goal_predicates_ids = convert_sample_to_pddl_goal_with_ids(ann,
                                                               pddl_goal_list,
                                                               relations_meta_ids,
                                                               obj_meta_ids)
    goal_predicates_with_id_names = convert_goal_id_to_names(id_to_name, goal_predicates_ids)
    general_goal_predicates = create_general_goal_predicates(goal_predicates_ids)

    if len(pddl_actions) == 0 or len(goal_predicates_ids) == 0:
        utils_variables.bad_tsk_id.append(ann['task_id'])
        return {'valid_meta': 'Non valid'}

    problem_path = problem_generator.create_basic_problem_template(this_problem_prefix + '_MetaData',
                                                                   obj_meta,
                                                                   relations_meta,
                                                                   general_goal_predicates)

    valid_add = check_if_solution_valid(pddl_actions, problem_path)

    if valid_add[0] != 'Valid':
        actions_with_ids, pddl_actions = goto_check(actions_with_ids, pddl_actions)
        valid_add = check_if_solution_valid(pddl_actions, problem_path)
        if valid_add[0] == 'Valid':
            print('valid after adding goto action')
            utils_variables.goto_action_added += 1

    sol_path = utils_paths.add + 'Data/{}/Solutions/{}_solution.soln'.format(valid_add[0], this_problem_prefix)
    write_solution(path=sol_path, actions=pddl_actions)
    #
    random.shuffle(relations_meta)
    random.shuffle(obj_meta)

    return {'obj_meta_ids': list(obj_meta_ids),
            'obj_meta': obj_meta,
            'relations_meta': relations_meta,
            'relations_meta_wo_ids': relations_reg_names,
            'pddl_actions_meta': pddl_actions,
            'actions_meta': actions_with_ids,
            'goal_meta': goal_predicates_with_id_names,
            'problem_path_meta': problem_path,
            'valid_meta': valid_add[0],
            'sol_path': sol_path}
