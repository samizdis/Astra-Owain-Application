from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import query_OAI_with_retries


def gpt_classify(labelled_examples: pd.DataFrame, test, model="gpt-4"):
    '''
    Given some labelled examples (row['text'] classified with row['label'] as True or False)
    Return the classification of 'test'
    '''

    messages=[]
    for i, row in labelled_examples.iterrows():
        messages.append({"role": "user", "content": row['text']})
        messages.append({"role": "assistant", "content": str(row['label'])})

    messages.append({"role": "user", "content": test})

    return query_OAI_with_retries(model=model, messages=messages).choices[0].message.content


def consecutive_classify(train_df: pd.DataFrame, test_df: pd.DataFrame, model: str):
    '''
    Iterate through test_df, classifing each element using the labelled train_df in-context
    '''

    responses = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), leave=False):
        responses.append(gpt_classify(
            labelled_examples=train_df,
            test=row['text'],
            model=model
        ))

    correct = test_df['label'].astype(str) == responses

    success = sum(correct) / len(correct)

    return responses, correct, success


def run_gpt_classification(balanced_dataset: pd.DataFrame, num_tests=20, max_num_examples=40, models_to_run=('gpt-4',)):

    test_frac = 0.2  # We subsample later, but this gives us disjoint sets
    train, test = train_test_split(balanced_dataset, test_size=test_frac, random_state=42)

    results = {}
    for model in models_to_run:
        print('Classifying using:', model)

        results[model] = []

        for num_examples in tqdm(np.arange(1, max_num_examples, step=5)):
            responses, correct, success = consecutive_classify(
                train.sample(n=num_examples),
                test.sample(n=num_tests),
                model=model
            )

            results[model].append({
                'num_examples': num_examples,
                'responses': responses,
                'correct': correct,
                'success': success,
            })

        results[model] = pd.DataFrame(results[model])

    return results


def gpt_explain_reasoning(labelled_examples: pd.DataFrame, model="gpt-4", system_prompt=None):
    '''
    What does 'model' return when asked to describe a classification rule for a dataset of labelled examples?
    '''

    if system_prompt is None:
        system_prompt = ("You are given some statements, which are labelled with categories. "
                         "Describe the classification rule.")

    messages=[
        {"role": "system",
         "content": system_prompt
        },
    ]
    for i, row in labelled_examples.iterrows():
        messages.append(
            {"role": "user",
             "content": row['text'] + ' |LABEL|: '+ str(row['label'])
            })

    return query_OAI_with_retries(model=model, messages=messages).choices[0].message.content


def gpt_classify_using_rule(test_statement: str, rule: str, model="gpt-4"):

    messages=[{
        "role": "system",
        "content": ("You are a helpful assistant. " +
                    "You classify statements, outputting only the classification label, "
                    "using the following rule: " +
                    rule)
    }]


    messages.append({"role": "user", "content": test_statement})

    return query_OAI_with_retries(model=model, messages=messages).choices[0].message.content


def consecutive_classify_using_rule(test_df: pd.DataFrame, rule: str, model):

    responses = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), leave=False):
        responses.append(gpt_classify_using_rule(
            test_statement=row['text'],
            rule=rule,
            model=model
        ))

    correct = test_df['label'].astype(str) == responses

    success = sum(correct) / len(correct)

    return responses, correct, success

