""" Converts csv into a format suitable for the q&a model.


Improvements:

"""


import re
import html
from pandas import DataFrame
import pandas as pd

from typing import  List


from utils.utils import *
from pathlib import Path


'''
filter hyperparameters

ignore rows with:
    - annotations
    - imgs (may contain code snippets)
    - code snippets

category

filtering regex: @|<img|<pre|\.png|def 

split multi answer questions into two separate rows

Right now not filtered by category. Can do this for stricter filtering.

#remove null values

TODO:


'''
def clean_csv(path_to_csv: str, save_path: str, filter_level=0) -> DataFrame:
    #relevant_fields = ['post_id', 'question_title', 'folders', 'question', 'student_answer', 'instructor_answer']
    relevant_fields = ['question_title','question', 'student_answer', 'instructor_answer']
    csv:DataFrame = pd.read_csv(path_to_csv)
    transform_csv = csv[relevant_fields]
    transform_csv.index.name = 'id'

    # replace null values with the empty string
    transform_csv =  transform_csv.fillna('')

    # concat question title and question together
    filled = (transform_csv['question'] != '') &  (transform_csv['question_title'] != '')
    transform_csv.loc[filled, 'question'] = transform_csv.loc[filled, 'question_title'] + ' ' + transform_csv.loc[filled, 'question'] 

    transform_csv.loc[transform_csv['question'] == '', 'question'] = transform_csv.loc[:, 'question_title']

    # get non-empty student and instructor answers
    student_answers = transform_csv.drop('instructor_answer', axis=1).rename(columns={'student_answer': 'answer'})
    student_answers = student_answers[student_answers['answer'] != '']
    instructor_answers = transform_csv.drop('student_answer', axis=1).rename(columns={'instructor_answer': 'answer'})
    instructor_answers = instructor_answers[instructor_answers['answer'] != '']
 
    transform_csv = pd.concat([student_answers, instructor_answers], ignore_index=True)
    

    # transform_csv = transform_csv.drop(['question_title', 'folders'], axis=1)
    transform_csv = transform_csv.drop(['question_title'], axis=1)

    # clean and filter question and answers

    transform_csv['question'] = transform_csv['question'].apply(html.unescape)
    transform_csv['answer'] =  transform_csv['answer'].apply(html.unescape)

    transform_csv = transform_csv[~transform_csv['question'].str.contains('@|<img|<pre|\.png|def ', flags=re.IGNORECASE)]
    transform_csv = transform_csv[~transform_csv['answer'].str.contains('@|<img|<pre|\.png|def ', flags=re.IGNORECASE)]

    transform_csv['question'] = transform_csv['question'].apply(strip_tags)
    transform_csv['answer'] =  transform_csv['answer'].apply(strip_tags)


    transform_csv =  transform_csv.fillna('')
    transform_csv =  transform_csv[transform_csv['question'] != '']
    transform_csv =  transform_csv[transform_csv['answer'] != '']
    
    #print(transform_csv)

    transform_csv.to_csv(save_path)

    return transform_csv


def combine_and_split(dataframes: List[DataFrame], train_pct, val_pct, save_dir, test_set:DataFrame=None):
    """
    Hold out an instance of 324 as test set
    """

    combined_datasets = pd.concat(dataframes, ignore_index=True).sample(frac=1, random_state=0)
    # size of all samples across train/val/test
    size = len(combined_datasets) 
    size += len(test_set) if isinstance(test_set, DataFrame) else 0
    print(f'total samples is: {size}')
    train, rest = np.split(combined_datasets, [np.floor(len(combined_datasets)* train_pct).astype(int)]) 
    if not isinstance(test_set, DataFrame): # further split rest
        val, test = np.split(rest, [np.floor(len(rest) * val_pct).astype(int)]) 
    else:
        val = rest
        test = test_set 


    # save to csv
    print(f'saving to {save_dir} directory')
    train.to_csv(save_dir + "train.csv")
    val.to_csv(save_dir + "val.csv")
    test.to_csv(save_dir + "test.csv")

    print(f'Number of train samples: {len(train)}')
    print(f'Number of val samples: {len(val)}')
    print(f'Number of test samples: {len(test)}')


def print_stats(dataset_path: str):
    p = Path(dataset_path)
    for file in p.iterdir():
        print(file.name)
        for idx, split in enumerate(['train', 'val', 'test']):
            if split in file.name: 
                dataset = pd.read_csv(file.absolute())
                print(len(dataset)) 