import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from . import utils_paths, utils_objects
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages


goal_per_type_precision = {}
goal_per_type_recall = {}

goal_general_precision = {}
goal_general_recall = {}

actions_per_type_precision = {}
actions_per_type_recall = {}

actions_general_precision = {}
actions_general_recall = {}

output_file_name = 'image_output.pdf'


def check_and_create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_dicts():
    return {'Per Type Precision': goal_per_type_precision,
            'Per Type Recall': goal_per_type_recall,
            'General Precision': goal_general_precision,
            'General Recall': goal_general_recall}


def create_pdf_with_img(figs, file_name='output.pdf', task_type='', pddl=''):
    check_and_create_dir('{}{}/{}'.format(utils_paths.results_path_pdf, task_type.upper(), pddl))
    file_name = '{}{}/{}{}'.format(utils_paths.results_path_pdf, task_type.upper(), pddl, file_name)
    if '.pdf' not in file_name:
        file_name = file_name + '.pdf'

    print('Saving Output file under the path {}'.format(file_name))
    pdf = PdfPages(file_name)
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()


def bars_from_dicts(dicts):
    plt.tight_layout()
    f, axes = plt.subplots(3, 2, figsize=(15, 15))
    for i, d in enumerate(dicts):
        sorted_d = sorted(d.items(), key=lambda item: item[1], reverse=True)
        df = pd.DataFrame(sorted_d, columns=['Type', 'Accuracy'])
        df['Accuracy'] = pd.to_numeric(df['Accuracy'], downcast='float')
        ax = sns.barplot(x='Type', y='Accuracy', data=df, palette='summer', ax=axes[i, 0])
        ax.set(ylim=(0, 1))

    create_pdf_with_img(f, output_file_name)


def create_multiple_cols(dicts, task_type):
    cols = 2
    rows = len(list(dicts.keys())) // 2

    axes_combinations = list(itertools.product(range(rows), range(cols)))
    f, axes = plt.subplots(rows, cols, figsize=(17, 16))
    for i, (model_name, d) in enumerate(dicts.items()):
        values = []
        for mode, dic in d.items():
            if 'per_value' not in model_name:
                order = utils_objects.pdf_order_dict[task_type]['general']
            else:
                order = utils_objects.pdf_order_dict[task_type]['per_value']

            for score_type in order:
                if score_type not in dic:
                    print('score type: {} not in dict: {}'.format(score_type, dic))
                else:
                    values.append([mode, score_type, dic[score_type]])

        new = pd.DataFrame(values, columns=['Mode', 'Class', 'Precision'])
        ax = sns.barplot(x='Class', y='Precision', hue='Mode', data=new, palette="rocket",
                         ax=axes[axes_combinations[i][0], axes_combinations[i][1]])
        ax.set_title(model_name, fontsize=20)
        ax.set(ylim=(0, 1.3))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
        ax.set(xlabel='', ylabel=model_name.split(' ')[-1])
        ax.axhline(1.0, ls='--')
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.setp(ax.get_legend().get_texts(), fontsize='7')
        plt.setp(ax.get_legend().get_title(), fontsize='7')

    return f


def df_to_pdf(dicts, task_type):
    cols = 1
    rows = 0
    for key, val in dicts.items():
        for _, _ in val.items():
            rows += 1

    f, axes = plt.subplots(rows, cols, figsize=(16, 16))
    j = -1
    for name, outer_dic in dicts.items():
        for mode, dic in outer_dic.items():
            j += 1

            acc_type = 'general' if 'general' in name.lower() else 'per_value'
            order_names = utils_objects.pdf_order_dict[task_type][acc_type]
            order_mapping = {key: i for i, key in enumerate(order_names)}
            dic = dict(sorted(dic.items(), key=lambda val_: order_mapping[val_[0]]))

            columns = list(dic.keys())
            values = list(dic.values())
            df = pd.DataFrame([values], columns=columns)
            c = df.shape[1]

            tb = axes[j].table(cellText=np.vstack([df.columns, df.values]),
                               cellColours=[['lightgray'] * c] + [['none'] * c],
                               bbox=[0, 0, 1, 1], cellLoc='center')

            tb.auto_set_font_size(False)
            tb.set_fontsize(12)

            axes[j].axis('off')
            axes[j].set_title('{}_{}'.format(name, mode),  fontsize=13)
            plt.subplots_adjust(hspace=0.75)

            for (row, col), cell in tb.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontproperties=FontProperties(weight='bold', size=12))

    return f
