import timeout_decorator
import pandas as pd

from openai import OpenAI

from config import OAI_keyfile

client = OpenAI(api_key=open(OAI_keyfile).read().strip())


@timeout_decorator.timeout(5)
def timecapped_query(model, messages):
    return client.chat.completions.create(model=model, messages=messages)


def query_OAI_with_retries(model, messages, retries=3):
    while True:
        if not retries:
            raise Exception('Out of retries')

        try:
            return timecapped_query(model, messages)

        except timeout_decorator.TimeoutError:
            retries -= 1


def save_dict_of_dfs_with_df_elements(dict_of_dfs, filename_stem):
    '''
    dict_of_dfs contains key:value like ('model': df)
    where rows of df['correct'] are also Dataframes which here we convert to string.
    '''
    for model, df in dict_of_dfs.items():
        df['correct'] = df['correct'].astype(str)
        df.to_feather(f'data/{filename_stem}-{model}.feather')

def load_dict_of_dfs_with_df_elements(filename_stem, models_to_load):
    '''
    load dict_of_dfs contains key:value like ('model': df)
    like the one saved above
    where rows of df['correct'] are also Dataframes which were converted to string.
    '''

    results_dict = {}

    for model in models_to_load:

        df = pd.read_feather(f'data/{filename_stem}-{model}.feather')

        # Turn from dfs-saved-as-text to dfs
        temp_dfs = []
        for row in df['correct']:
            split_lines = [
                line.strip().split()
                for line in row.split('\n')
                if line.strip() != ''
                and not line.startswith('Name:')
            ]
            temp_dfs.append(pd.DataFrame(split_lines, columns=['Index', 'Value']))
        df['correct'] = temp_dfs

        results_dict[model] = df

    return(results_dict)


