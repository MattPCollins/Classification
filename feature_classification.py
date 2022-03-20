import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import itertools
import random
import re
import plotly.express as px
from pprint import pprint
pd.options.mode.chained_assignment = None  # default='warn'
np.set_printoptions(precision=3)



# Use SequenceMatcher function to calculate string similarity
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Create correlation matrix figure
def correlation_matrix(titles):
    df_list = []
    for i in titles:
        row = [i]
        for j in titles:
            row.append(similar(j,i))
        df_list.append(row)

    df_x = pd.DataFrame(df_list, columns = ['title'] + titles)
    df_x.set_index('title', inplace=True)

    fig = px.imshow(df_x)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    return df_x, fig


# Apply grouping and return "stripped down" version of titles using the common values identified in the similarity check
def group_similar(df,percent):
    s = np.where(df.gt(percent,0), [i for i in df.columns], '') # Note this doesn't yet work with pyspark.pandas!
    similar_lists = pd.Series([list(filter(None, i)) for i in s])
    
    groups = []
    for i in similar_lists:
        groups.append(list(set(i)))
    groups.sort()
    groups = list(groups for groups,_ in itertools.groupby(groups))
    
    remove_ = [group for group in groups if len(group) == 1]
    for i in remove_:
        groups.remove(i)

    # https://stackoverflow.com/questions/38862657/find-value-greater-than-level-python-pandas
    # Add comment on this function found online
    def long_substr(data):
        substrs = lambda x: {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}
        s = substrs(data[0])
        for val in data[1:]:
            s.intersection_update(substrs(val))
        return max(s, key=len)

    stripped = []
    for i in groups:
        stripped.append(long_substr(i))
    stripped = [i.strip() for i in stripped]
    stripped = list(set(stripped))
    
    # Add removed titles back in
    removed = [i[0] for i in remove_]
    titles_stripped = [*removed, *stripped]
    titles_stripped = list(set(titles_stripped))    
    return titles_stripped

# Evaluate the independence between each of the stripped titles.
def independence_metric(titles_stripped):
    independence = []
    for i in titles_stripped:
        other_titles = [j for j in titles_stripped if j != i]
        independent = 0
        for j in other_titles:
            if i not in j and len(i) > 1: 
                independent += 1
        if len(other_titles) == 0:
            val = 0
        else:
            val = round(independent/len(other_titles),2)
        independence.append(val)
    mean_independence = np.mean(independence)
    return mean_independence


# Main function to bring everything together
def full(titles, percent=0.8,levels=1):
    features = [len(titles)]
    counter = 0 
    titles_stripped = titles
    while counter < levels:
        df, fig = correlation_matrix(titles_stripped)
        titles_stripped = group_similar(df,percent)
        features.append(len(titles_stripped))
        counter += 1
        pprint(titles_stripped)
        mean_independence = independence_metric(titles_stripped)
    return titles_stripped, features, fig, mean_independence


# Test different similarities
def test_thresholds(titles, thresholds):
    #thresholds = list(range(1,10))
    independences = []
    for i in thresholds:
        print('Similarity: {}%'.format(i * 10))
        titles_stripped, features, fig, mean_independence = full(titles, 0.1 * i, 1)
        print(features)
        print('Mean independence: {}'.format(round(mean_independence,2)))
        independences.append(round(mean_independence,2))
        print()
    fig.show()
    #fig.write_html("path/to/file.html")
    return independences
    
def plot_thresholds(thresholds, independences):
    thresholds_percent = [i * 10 for i in thresholds]
    independences_percent = [i * 100 for i in independences]
    fig = px.line(
        x=thresholds_percent,
        y=independences_percent,
        labels=dict(x="Similarity (%)", y="Mean independence of values (%)"))
    fig.show()
    
    
# Substitute the stripped down titles into the DataFrame.
def substitute_to_df(df, column, stripped):
    list_1 = []
    for i in df[column]:
        list_2 = []
        for j in stripped:
            if j in i.lower():
                list_2.append(j)
            else:
                pass
        if list_2:
            list_1.append(list_2[0])
        else:
            list_1.append(i)
    col_stripped = column + '_stripped'
    df[col_stripped] = list_1
    return df

