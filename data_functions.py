import pandas as pd


def balanced_dataset(df: pd.DataFrame, test_fn, size=None, shuffled=True, random_state=42):
    '''
    Run test_fn on df, which should classify df['text'] to either True or False
    Return a dataset with an equal number of Truthy and Falsey rows
    '''

    if size is None:  # As big as possible
        num_true = sum(df['text'].apply(test_fn) == True)
        num_false = sum(df['text'].apply(test_fn) == False)
        size = 2 * min(num_false, num_true)

    ds = pd.concat([
        df[df['text'].apply(test_fn) == True].sample(size//2, random_state=random_state),
        df[df['text'].apply(test_fn) == False].sample(size//2, random_state=random_state),
    ])

    return ds.sample(frac=1, random_state=random_state) if shuffled else ds

