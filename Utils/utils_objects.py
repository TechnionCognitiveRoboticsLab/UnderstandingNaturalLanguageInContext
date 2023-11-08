all_obj_from_ai2thor = ['alarmclock',
                        'aluminumfoil',
                        'apple',
                        'applesliced',
                        'armchair',
                        'baseballbat',
                        'basketball',
                        'bathtub',
                        'bathtubbasin',
                        'bed',
                        'blinds',
                        'book',
                        'boots',
                        'bottle',
                        'bowl',
                        'box',
                        'bread',
                        'breadsliced',
                        'butterknife',
                        'cabinet',
                        'candle',
                        'cart',
                        'cd',
                        'cellphone',
                        'chair',
                        'cloth',
                        'coffeemachine',
                        'coffeetable',
                        'countertop',
                        'creditcard',
                        'cup',
                        'curtains',
                        'desk',
                        'desklamp',
                        'desktop',
                        'diningtable',
                        'dishsponge',
                        'dogbed',
                        'drawer',
                        'dresser',
                        'dumbbell',
                        'egg',
                        'eggcracked',
                        'faucet',
                        'floor',
                        'floorlamp',
                        'footstool',
                        'fork',
                        'fridge',
                        'garbagebag',
                        'garbagecan',
                        'glassbottle',
                        'handtowel',
                        'handtowelholder',
                        'houseplant',
                        'kettle',
                        'keychain',
                        'knife',
                        'ladle',
                        'lamp',
                        'laptop',
                        'laundryhamper',
                        'laundryhamperlid',
                        'lettuce',
                        'lettucesliced',
                        'lightswitch',
                        'microwave',
                        'mirror',
                        'mug',
                        'newspaper',
                        'ottoman',
                        'painting',
                        'pan',
                        'papertowelroll',
                        'pen',
                        'pencil',
                        'peppershaker',
                        'pillow',
                        'plate',
                        'plunger',
                        'poster',
                        'pot',
                        'potato',
                        'potatosliced',
                        'remotecontrol',
                        'roomdecor',
                        'safe',
                        'saltshaker',
                        'scrubbrush',
                        'shelf',
                        'shelvingunit',
                        'showercurtain',
                        'showerdoor',
                        'showerglass',
                        'showerhead',
                        'sidetable',
                        'sink',
                        'sinkbasin',
                        'soapbar',
                        'soapbottle',
                        'sofa',
                        'spatula',
                        'spoon',
                        'spraybottle',
                        'statue',
                        'stool',
                        'stoveburner',
                        'stoveknob',
                        'tabletopdecor',
                        'targetcircle',
                        'teddybear',
                        'television',
                        'tennisracket',
                        'tissuebox',
                        'toaster',
                        'toilet',
                        'toiletpaper',
                        'toiletpaperhanger',
                        'tomato',
                        'tomatosliced',
                        'towel',
                        'towelholder',
                        'tvstand',
                        'vacuumcleaner',
                        'vase',
                        'watch',
                        'wateringcan',
                        'window',
                        'winebottle']

actions_dict = {'pickupobject': 'pick up',
                'gotolocation': 'go to',
                'putobject': 'put',
                'sliceobject': 'slice',
                'heatobject': 'heat',
                'coolobject': 'cool',
                'cleanobject': 'clean',
                'toggleobject': 'toggle',
                'noop': 'noop',
                'end': 'end'}

receptacles = ['fridge', 'microwave', 'safe', 'sink', 'pot', 'bathtubbasin', 'bowl', 'box', 'cabinet',
               'cart', 'cup', 'drawer', 'garbagecan', 'mug', 'pan', 'toilet', 'stoveburner', 'dresser',
               'kettle', 'diningtable', 'sidetable', 'shelf', 'sinkbasin', 'desklamp', 'sofa',
               'countertop', 'bed', 'desk', 'coffeetable', 'coffeemachine', 'armchair', 'toiletpaperhanger',
               'handtowelholder', 'ottoman']

receptacles_non_mobile = ['fridge', 'microwave', 'safe', 'sink', 'bathtubbasin', 'cabinet', 'cart', 'drawer', 'toilet',
                          'stoveburner', 'dresser', 'diningtable', 'sidetable', 'shelf', 'sinkbasin', 'sofa',
                          'countertop', 'bed', 'desk', 'coffeetable', 'armchair', 'toiletpaperhanger',
                          'handtowelholder', 'ottoman']

receptacles_mobile = ['pot',  'bowl', 'box', 'cup', 'garbagecan', 'mug', 'pan', 'kettle', 'desklamp', 'coffeemachine']

in_not_on = ['fridge', 'microwave', 'safe', 'sinkbasin', 'pot', 'bathtubbasin', 'bowl', 'box', 'cabinet', 'cart',
             'cup', 'drawer', 'garbagecan', 'mug', 'pan', 'pot', 'safe', 'sinkbasin', 'toilet']


plan_example = ['walk three steps towards the dining table . '
                 'turn left and face the table, pick up the knife from the table . '
                 'on the cabinet there is a tomato, take it . '
                 'cut the tomato into multiple pieces . '
                 'put the tomato in the bin .']

plan_examples = ['task: To heat a slice of bread and place it on the counter to the right of the stove., '
                 'NoisedRelations: on potato microwave , on vase microwave , on soapbottle mug , on knife countertop , '
                 'on cup microwave , on tomato microwave , on glassbottle microwave , on bread countertop , '
                 'on spoon microwave , on apple microwave , on papertowelroll pot , on cellphone cup , '
                 'on plate microwave , on statue microwave , on peppershaker microwave , on creditcard microwave , '
                 'on dishsponge microwave , on spatula microwave, '
                 'objects: countertop , potato , creditcard , '
                 'cellphone , dishsponge , butterknife , spatula , knife , egg , cup , plate , tomato , '
                 'soapbottle , bowl , saltshaker , peppershaker , bread , statue , glassbottle , fork , '
                 'pan , lettuce , apple , pot , papertowelroll , vase , spoon , mug , microwave']

can_cook_objects = ["microwave", "stoveburner"]

can_open_objects = ["safe", "microwave", "drawer", "cabinet", "fridge", "laptop", "box"]

can_be_sliced_objects = ["apple", "bread", "lettuce", "potato", "tomato"]

can_turn_on_objects = ["microwave", "desklamp", "floorlamp"]

can_cut_objects = ["knife", "butterknife"]

can_contain_objects = ["bathtubbasin", "bowl", "box", "cabinet", "cart", "drawer", "dresser", "fridge",
                       "garbagecan", "kettle", "microwave", "mug", "pot", "sink", "vase", "cup", "pan"]

can_wash_objects = ['sink', "bathtubbasin", 'sinkbasin']

can_cool_objects = ['fridge']

obj_features = {'can_cook': can_cook_objects,
                'can_open': can_open_objects,
                'can_be_sliced': can_be_sliced_objects,
                'can_turn_on': can_turn_on_objects,
                'can_cut': can_cut_objects,
                'can_contain': can_contain_objects,
                'can_wash': can_wash_objects,
                'can_cool': can_cool_objects}


common_two_words_objects = {'butter knife': 'butterknife',
                            'floor lamp': 'floorlamp',
                            'desk lamp': 'desklamp',
                            'remote control': 'remotecontrol',
                            'credit card': 'creditcard',
                            'dish sponge': 'dishsponge',
                            'bathtub basin': 'bathtubbasin',
                            'side table': 'sidetable',
                            'tissue box': 'tissuebox',
                            'coffee machine': 'coffeemachine',
                            'garbage can': 'garbagecan',
                            'hand towel holder': 'handtowelholder',
                            'toilet paper hanger': 'toiletpaperhanger',
                            'toilet paper': 'toiletpaper',
                            'alarm clock': 'alarmclock',
                            'dining table': 'diningtable',
                            'coffee table': 'coffeetable',
                            'hand towel': 'handtowel',
                            'spray bottle': 'spraybottle',
                            'wine bottle': 'winebottle',
                            'stove burner': 'stoveburner',
                            'sink': 'sinkbasin',
                            'glass bottle': 'glassbottle',
                            'soap bottle': 'soapbottle',
                            'baseball bat': 'baseballbat',
                            'pepper shaker': 'peppershaker',
                            'salt shaker': 'saltshaker',
                            'tennis racket': 'tennisracket',
                            'watering can': 'wateringcan',

                            'go to ': 'gotolocation ',
                            'pick up ': 'pickupobject ',
                            'goto ': 'gotolocation ',
                            'pickup ': 'pickupobject ',
                            'toggle': 'toggleobject',
                            'heat': 'heatobject',
                            'clean': 'cleanobject',
                            'cool': 'coolobject',
                            'put': 'putobject'}


# common_two_words_objects_gpt = {'butter knife': 'butterknife',
#                                 'floor lamp': 'floorlamp',
#                                 'desk lamp': 'desklamp',
#                                 'remote control': 'remotecontrol',
#                                 'credit card': 'creditcard',
#                                 'dish sponge': 'dishsponge',
#                                 'soap bar': 'soapbar',
#                                 'key chain': 'keychain',
#                                 'paper shaker': 'papershaker',
#                                 'salt mill': 'saltmill',
#                                 'salt container': 'saltcontainer',
#                                 'foot rest': 'footrest',
#                                 'bathtub basin': 'bathtubbasin',
#                                 'side table': 'sidetable',
#                                 'tissue box': 'tissuebox',
#                                 'coffee machine': 'coffeemachine',
#                                 'garbage can': 'garbagecan',
#                                 'hand towel holder': 'handtowelholder',
#                                 'toilet paper hanger': 'toiletpaperhanger',
#                                 'toilet paper': 'toiletpaper',
#                                 'tissue paper': 'tissuepaper',
#                                 'alarm clock': 'alarmclock',
#                                 'dining table': 'diningtable',
#                                 'coffee table': 'coffeetable',
#                                 'hand towel': 'handtowel',
#                                 'spray bottle': 'spraybottle',
#                                 'wine bottle': 'winebottle',
#                                 'stove burner': 'stoveburner',
#                                 'sink basin': 'sinkbasin',
#                                 'glass bottle': 'glassbottle',
#                                 'soap bottle': 'soapbottle',
#                                 'baseball bat': 'baseballbat',
#                                 'pepper shaker': 'peppershaker',
#                                 'salt shaker': 'saltshaker',
#                                 'tennis racket': 'tennisracket',
#                                 'watering can': 'wateringcan',
#                                 # 'go to ': 'goto ',
#                                 # 'pick up ': 'pickup ',
#                                 'bath tub': 'bathtub',
#                                 ',': '.'}


action_correct_dict = {'command': 0,
                       'arg1': 0,
                       'arg2': 0,
                       'p_arg1': 0,
                       'p_arg2': 0,
                       'full_triple': 0,
                       'full_seq': 0,
                       'full_minus1': 0,
                       'full_pddl': 0,
                       'full_pddl_pred_goal': 0}


goal_correct_dict = {'predicate_type': 0,
                     'arg1': 0,
                     'arg2': 0,
                     'p_arg1': 0,
                     'p_arg2': 0,

                     'f_predicate': 0,
                     'f_predicate_sim': 0,

                     'f_seq': 0,
                     'f_seq_sim': 0}


total_counter = {'triples_pred': 0,
                 'triples_true': 0,
                 'sequences': 0,
                 'arg2_pred': 0,
                 'arg2_true': 0,

                 'go to_true': 0,
                 'go to_pred': 0,

                 'pick up_true': 0,
                 'pick up_pred': 0,

                 'toggle_true': 0,
                 'toggle_pred': 0,

                 'slice_true': 0,
                 'slice_pred': 0,

                 'put_true': 0,
                 'put_pred': 0,

                 'heat_true': 0,
                 'heat_pred': 0,

                 'cool_true': 0,
                 'cool_pred': 0,

                 'clean_true': 0,
                 'clean_pred': 0}


total_counter_goal = {'predicates_pred': 0,
                      'predicates_true': 0,
                      'sequences': 0,

                      'arg1_pred': 0,
                      'arg1_true': 0,

                      'arg2_pred': 0,
                      'arg2_true': 0,

                      'toggled_true': 0,
                      'toggled_pred': 0,

                      'robot_has_obj_true': 0,
                      'robot_has_obj_pred': 0,

                      'cleaned_true': 0,
                      'cleaned_pred': 0,

                      'sliced_true': 0,
                      'sliced_pred': 0,

                      'on_true': 0,
                      'on_pred': 0,

                      'cold_true': 0,
                      'cold_pred': 0,

                      'hot_true': 0,
                      'hot_pred': 0,

                      'two_task_true': 0,
                      'two_task_pred': 0}


# total_counter_goal = {'predicates_pred': 0,
#                       'predicates_true': 0,
#                       'sequences': 0,
#                       'arg2_pred': 0,
#                       'arg2_true': 0,
#
#                       'toggle_true': 0,
#                       'toggle_pred': 0,
#
#                       'has_true': 0,
#                       'has_pred': 0,
#
#                       'clean_true': 0,
#                       'clean_pred': 0,
#
#                       'sliced_true': 0,
#                       'sliced_pred': 0,
#
#                       'on_true': 0,
#                       'on_pred': 0,
#
#                       'cold_true': 0,
#                       'cold_pred': 0,
#
#                       'hot_true': 0,
#                       'hot_pred': 0}

similar_objects = {'sink': ['sinkbasin', 'sink'],
                   'sinkbasin': ['sink', 'sinkbasin'],

                   'bathtub': ['bathtubbasin', 'bathtub'],
                   'bathtubbasin': ['bathtub', 'bathtubbasin'],

                   'knife': ['butterknife', 'knife'],
                   'butterknife': ['knife', 'butterknife'],

                   'mug': ['cup', 'mug'],
                   'cup': ['mug', 'cup'],

                   'lamp': ['desklamp', 'floorlamp', 'lamp'],
                   'floorlamp': ['desklamp', 'lamp', 'floorlamp'],
                   'desklamp': ['floorlamp', 'lamp', 'desklamp']}

del_w = [' in the ', ' on the ', ' on ', ' the ', ' <arg1> ', ' <arg2> ', 'iphone, ']

pdf_order_dict = \
{'actions':
  {'general': ['command', 'arg1', 'p_arg1', 'arg2', 'p_arg2', 'full_triple', 'full_seq', 'full_minus1', 'full_pddl',
               'full_pddl_pred_goal'],
   'per_value': ['goto', 'pickup', 'put', 'slice', 'toggle', 'cool', 'clean', 'heat']},
 'goal':
  {'general': ['predicate_type', 'arg1', 'p_arg1', 'arg2', 'p_arg2',
               'f_predicate', 'f_predicate_sim', 'f_seq', 'f_seq_sim'],
   'per_value': ['robot_has_obj', 'on', 'sliced', 'toggled', 'cleaned', 'cold', 'hot', 'two_task']}}


common_goal_mistakes = ['goal', ':']
