import re
import random
from collections import defaultdict
from Utils import utils_variables, utils_functions


def get_scene_data_from_js(ann, actions):
    all_objects_in_scene = defaultdict(int)
    all_objects_relation = set()

    object_poses = ann['scene']['object_poses']
    plan = ann['plan']['high_pddl']

    parent = ann['pddl_params']['parent_target'].lower()
    all_objects_in_scene[parent] = 0

    for pos in object_poses:
        obj = pos['objectName'].split('_')[0].lower()
        all_objects_in_scene[obj] += 1

    for action_box in plan:
        discrete = action_box['discrete_action']
        planner = action_box['planner_action']

        if 'coordinateReceptacleObjectId' in planner:
            receptacle_obj = planner['coordinateReceptacleObjectId'][0].lower()

            if receptacle_obj not in all_objects_in_scene:
                all_objects_in_scene[receptacle_obj] += 1

            if discrete['action'].lower() in ['pickupobject']:
                main_obj = planner['coordinateObjectId'][0].lower()
                if ' {} '.format(main_obj) not in ' '.join(all_objects_relation) \
                        or (all_objects_in_scene[main_obj] > 1 and 'two' in ann['task_type']):
                    all_objects_relation.add('on {} {}'.format(main_obj, receptacle_obj))

    if 'slice' in actions[-1]:
        if actions[-1].split(' ')[2] not in ' '.join(all_objects_relation):
            sliced_obj_relation = 'on {} {}'.format(actions[-1].split(' ')[2],
                                                    ann['pddl_params']['parent_target'].lower())
            all_objects_relation.add(sliced_obj_relation)

    if '' in all_objects_in_scene:
        del all_objects_in_scene['']
    return list(all_objects_in_scene.keys()), all_objects_relation


def get_recept_from_metadata(obj, metadata, param):
    param = list(set(utils_functions.extract_obj_id(p) for p in param))
    param_only_name = [x.split('|')[0] for x in param]
    for special_obj in ['sink', 'bathtub']:
        if special_obj in param_only_name and '{}basin'.format(special_obj) in param_only_name:
            param = [x for x in param if x.split('|')[0] != special_obj]

    possible_recept = []
    for obj1 in param:
        parent = False
        objects_inside = list(set(utils_functions.extract_obj_id(p) for p in metadata[obj1]['receptacleObjectIds']))
        for obj2 in param:
            if obj2 in objects_inside:
                parent = True
                break
        if not parent and obj in objects_inside:
            possible_recept.append(utils_functions.extract_obj_id(obj1))
    if len([x for x in possible_recept if 'stoveburner' not in x]) > 1 and obj.split('|')[0] not in \
            ['faucet', 'showerglass', 'showercurtain', 'sink', 'tvstand', 'houseplant']:
        utils_variables.more_than_one_recept_in_metadata += 1
    return possible_recept


def get_scene_info_from_metadata(metadata):
    all_objects = defaultdict(int)
    all_dup_names = set()
    all_objects_with_id = set()
    loc_id_to_num_id = {}
    relations = []
    relations_with_full_id = []
    relations_reg_names = []

    metadata_id_dict = {utils_functions.extract_obj_id(y['objectId']): y for x, y in metadata.items()}

    keys_list = list(metadata.keys())
    random.shuffle(keys_list)

    obj_used_ids = {}

    for obj in keys_list:
        obj_id_name = utils_functions.extract_obj_id(metadata[obj]['objectId'])
        obj_name = obj_id_name.split('|')[0]

        obj_name_key = obj.split('_')[0].lower()
        if obj_name_key in ['sink', 'bathtub']:
            if obj_name_key + 'basin' in ' '.join(keys_list).lower():
                continue
        all_objects[obj_name] += 1

        # generate random id
        rand_id = random.randint(100, 999)
        if obj_name not in obj_used_ids:
            obj_used_ids[obj_name] = set()
        else:
            while rand_id in obj_used_ids[obj_name]:
                rand_id = random.randint(100, 999)

        obj_used_ids[obj_name].add(rand_id)
        obj_dup_name = '{}{}'.format(obj_name, rand_id).lower()

        all_dup_names.add(obj_dup_name)
        all_objects_with_id.add(obj_id_name)
        loc_id_to_num_id[obj_id_name] = obj_dup_name

    all_dup_names = list(all_dup_names)

    for obj, data in metadata.items():
        obj_id = utils_functions.extract_obj_id(data['objectId'])
        obj_dup_name = loc_id_to_num_id[obj_id]

        if data['parentReceptacles'] is not None:
            if len(data['parentReceptacles']) == 1:
                obj_recept_list = [utils_functions.extract_obj_id(data['parentReceptacles'][0])]
            else:
                obj_recept_list = get_recept_from_metadata(obj_id, metadata_id_dict, data['parentReceptacles'])
            for recept_id in obj_recept_list:
                recep_name = loc_id_to_num_id[recept_id]
                obj_reg_name = re.match('[a-zA-Z]+', obj_dup_name).group(0)
                recep_reg_name = re.match('[a-zA-Z]+', recep_name).group(0)

                relations.append('on {} {}'.format(obj_dup_name, recep_name))
                relations_reg_names.append('on {} {}'.format(obj_reg_name, recep_reg_name))
                relations_with_full_id.append('on {} {}'.format(obj_id, recept_id))

    all_dup_names.sort()
    relations.sort()
    name_to_id_dict = {val: key for key, val in loc_id_to_num_id.items()}

    return all_objects_with_id, all_dup_names, relations, relations_with_full_id, loc_id_to_num_id, name_to_id_dict, \
           relations_reg_names
