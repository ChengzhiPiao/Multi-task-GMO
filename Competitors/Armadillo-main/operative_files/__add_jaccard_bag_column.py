"""Most of the code in this file was taken from https://gist.github.com/kawa-kokosowa/59da8e964bdc26f46ee3e3164c6a8b4d"""
import pandas as pd
import pickle
from tqdm import tqdm
"""Jaccard Similarity of Bags
Using artificial restriciton of using only builtin Python 3 only.
"""

import doctest
import time
import itertools

def multiset_intersection_cardinality(x: list, y: list) -> int:
    """Returns the number of elements of x and y intersection."""

    cardinality = 0
    fewest, most = (x, y) if len(x) < len(y) else (y, x)
    most = most.copy()
    for value in fewest:
        try:
            most.remove(value)
        except ValueError:
            pass
        else:
            cardinality += 1

    return cardinality

def multiset_union_cardinality(x: list, y: list) -> int:
    """Return the number of elements in both x and y."""
    return len(x) + len(y)

def multiset_smaller_cardinality(x: list, y: list) -> int:
    """Return the number of elements in both x and y."""
    return min(len(x), len(y))

def jaccard_similarity_bags(x: list, y: list) -> float:
    """Get the Jaccard similarity of two bags (aka multisets).
    Example:
        >>> jaccard_similarity_bags([1,1,1,2], [1,1,2,2,3])
        0.3333333333333333
        >>> jaccard_similarity_bags([1,1,1,2], [1,2,3,4])
        0.25
        >>> jaccard_similarity_bags([1,1,2,2,3], [1,2,3,4])
        0.3333333333333333
    """
    intersection_cardinality = multiset_intersection_cardinality(x, y)
    union_cardinality = multiset_union_cardinality(x, y)
    smaller_cardinality = multiset_smaller_cardinality(x,y)
    return 2*(intersection_cardinality / union_cardinality), (intersection_cardinality / smaller_cardinality)

def get_value_list(df):
    values = df.values.tolist()
    out = list(itertools.chain.from_iterable(values))
    return out

def add_jaccard_bag_column(df: str|pd.DataFrame, table_dict: dict|str):
    print('Loading triple file')
    if isinstance(df, str):
        df = pd.read_csv(df)
    print('Loading table dict')
    if isinstance(table_dict, str):
        with open(table_dict, 'rb') as f:
            table_dict = pickle.load(f)
    new_values_union = []
    new_values_smaller = []
    new_times = []
    for r in tqdm(range(df.shape[0])):
        sample = df.iloc[r]
        start = time.time()
        r_table = get_value_list(table_dict[sample.loc['r_id']])
        s_table = get_value_list(table_dict[sample.loc['s_id']])
        int_over_union, int_over_smaller = jaccard_similarity_bags(r_table, s_table)
        end = time.time()
        new_times.append(end-start)
        new_values_union.append(int_over_union)
        new_values_smaller.append(int_over_smaller)
        
    df['jaccard_bag_union_times'] = new_times
    df['jaccard_bag_union_predictions'] = new_values_union
    df['jaccard_bag_union_ae'] = abs(df['jaccard_bag_union_predictions']-df['a%'])

    df['jaccard_bag_smaller_times'] = new_times
    df['jaccard_bag_smaller_predictions'] = new_values_smaller
    df['jaccard_bag_smaller_ae'] = abs(df['jaccard_bag_smaller_predictions']-df['a%']
                                       )
    return df
    
if __name__ == '__main__':
    wikitables = {
        'df':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/evaluation/eval_wiki.csv',
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl'
    }
    gittables = {
        'df':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/evaluation/eval_wiki.csv',
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/dictionaries/table_dictionaries/table_dict.pkl'
    }
    table_querying = {
        'df':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/evaluation/table_querying_results.csv',
        'table_dict':'/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/dictionaries/table_dict_table_querying.pkl'
    }
    df_wiki = add_jaccard_bag_column(**wikitables)
    df_wiki.to_csv('/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/evaluation/eval_wiki.csv', index=False)
    df_git = add_jaccard_bag_column(**gittables)
    df_git.to_csv('/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/evaluation/eval_wiki.csv', index=False)
    df_table_querying = add_jaccard_bag_column(**table_querying)
    df_table_querying.to_csv('/home/francesco.pugnaloni/armadillo_all/datasets/GitTables/table_querying/evaluation/table_querying_results.csv', index=False)
    pass

