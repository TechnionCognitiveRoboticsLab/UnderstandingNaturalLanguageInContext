import re
from random import randrange
from Utils import utils_paths, utils_variables, utils_objects


def create_objects_list(objects_list):
    return ['{} - object'.format('_'.join(obj.split(' '))) for obj in objects_list]


def create_objects_list_with_types(objects_list):
    obj_list_result = []
    for obj in objects_list:
        obj_type = re.findall("[a-zA-Z]+", obj)[0]
        if obj_type in utils_objects.all_obj_from_ai2thor:
            obj_list_result.append('{} - {}'.format(obj, obj_type))
        else:
            print('obj type {} not in ai2thor documentation'.format(obj_type))
            obj_list_result.append('{} - object'.format(obj, obj_type))
    return obj_list_result


def generate_random_object(objects_list):
    obj_index = randrange(len(objects_list))
    return objects_list[obj_index]


def simple_predicate_for_goal(predicate_name, obj, sign):
    if sign != '~':
        return '({} {})'.format(predicate_name, obj)
    return '(not ({} {}))'.format(predicate_name, obj)


def multiple_obj_predicate_for_goal(predicate_name, obj1, obj2, sign=''):
    if sign != '~':
        return '({} {} {})'.format(predicate_name, obj1, obj2)
    return '(not ({} {}))'.format(predicate_name,  obj1, obj2)


def create_features_predicates(features, objects_in_scene):
    predicates = []
    for feature, obj_list in features.items():
        for obj in obj_list:
            if utils_variables.use_meta:
                for scene_obj in objects_in_scene:
                    obj_regex = re.match("[a-zA-Z]+", scene_obj).group(0)
                    if obj == obj_regex:
                        predicates.append('{} {}'.format(feature, '_'.join(scene_obj.split(' '))))

            elif obj in objects_in_scene:
                predicates.append('{} {}'.format(feature, '_'.join(obj.split(' '))))

    return predicates


def create_basic_problem_template(problem_name,
                                  objects_list,
                                  objects_relations,
                                  goal_predicates,
                                  and_or="and"):

    PDDL_OBJECTS = create_objects_list_with_types(objects_list)
    OBJECTS_FEATURES = create_features_predicates(utils_objects.obj_features, objects_list)

    with open('{}{}.pddl'.format(utils_paths.problem_path, problem_name), 'w') as f:

        lines = ['(define\n',
                 '\t(problem {})\n'.format(problem_name),
                 '(:domain {})\n'.format(utils_variables.DOMAIN_NAME),
                 '\n(:objects\n']

        f.writelines(lines)

        for obj in PDDL_OBJECTS:
            f.write('{}\n'.format(obj))
        f.write(')\n')

        f.write('(:init\n')

        for feature in OBJECTS_FEATURES:
            f.write('({})\n'.format(feature))

        for feature in objects_relations:
            f.write('({})\n'.format(feature))

        f.write('(can_move)\n')
        f.write('(arm_free)\n')
        f.write(')\n')

        f.write('\n(:goal ({}\n'.format(and_or))
        for goal_pred in goal_predicates:
            f.write('{}\n'.format(goal_pred))
        f.write(')))\n')

    return '{}{}.pddl'.format(utils_paths.problem_path, problem_name)


def create_problem(problem_name,
                   object_list,
                   objects_relations,
                   obj,
                   goal_predicates=(),
                   receptacle=None):

    action_dict = {'clean': 'clean', 'heat': 'hot', 'cool': 'cold'}

    if len(goal_predicates) == 0:
        goal_predicates = []

        if receptacle:
            if receptacle in utils_objects.in_not_on:
                receptacle_predicate = multiple_obj_predicate_for_goal("in", obj, receptacle)
            elif 'examine' not in problem_name:
                receptacle_predicate = multiple_obj_predicate_for_goal("on", obj, receptacle)
            else:
                receptacle_predicate = '(toggled {})'.format(receptacle)
                goal_predicates.append('(robot_has_obj {})'.format(obj))
                goal_predicates.append('(can_reach {})'.format(receptacle))

            goal_predicates.append(receptacle_predicate)

        if 'clean' in problem_name or 'heat' in problem_name or 'cool' in problem_name:
            goal_predicates.append('({} {})'.format(action_dict[problem_name.split('_')[0]], obj))

    create_basic_problem_template(problem_name, object_list, objects_relations, goal_predicates)
