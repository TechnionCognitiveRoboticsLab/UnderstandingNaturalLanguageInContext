from collections import defaultdict

use_meta = True
use_cuda = True
use_multiprocessing = True

# ------------------------------------------------------------------------
# Load from pretrained
load_model = False

# Preprocess Data
remove_duplicates = True
preprocess_data = True
validate_data = False
benchmark_data = False

# Find lengths of data
find_lengths = True

# Train the model
train_model = True

# Evaluate the model on train, val_seen and val_unseen
eval_all = True

# Evaluate the model on val_seen and val_unseen during training
eval_val = False

# Generate predictions from unsupervised text
predict_list = False

# Return multiple sentences as output
multiple_sentences = True

# Validate the plans with PDDL
pddl_validator = False

# Create output PDF
create_output_pdf = True
# ------------------------------------------------------------------------

DOMAIN_NAME = 'ALFRED_World'

# Preprocessing variables
bad = 0
bad_goal = 0
total_anns = 0
added_relation = 0
goto_action_added = 0
different_goal_len = 0
no_recept_in_pickup = 0
no_arg_in_heat_cool = 0
direct_object_command = 0
pickup_only_one_actions = 0
more_than_one_recept_in_metadata = 0

bad_tsk_id = []
noises = [x/5 for x in range(1, 6)]
missing_main_obj = defaultdict(int)
missing_receptacle_obj = defaultdict(int)

action_prediction_prefix = 'translate plan to actions'
goal_prediction_prefix = 'translate task to goal'
prefixes = {'goal': goal_prediction_prefix, 'actions': action_prediction_prefix}

kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

all_scenes_names = {'kitchens': kitchens,
                    'living_rooms': living_rooms,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms}

scene_type_dic = {}
for scene, names in all_scenes_names.items():
    for name in names:
        scene_type_dic[name] = scene
