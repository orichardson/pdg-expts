
#%%
import sys; sys.path.append("../..")

from matplotlib import pyplot as plt
import numpy as np
import json

with open('datapts-all.json', 'r') as f:
    data = json.load(f)

#%%
from pdg.store import TensorLibrary

lib = TensorLibrary()

pt = data[0]
pt
# del pt['input_stats']
# lib(**pt)

# for pt in data:

import pandas as pd
df = pd.DataFrame(data)
# df.drop(['input_stats', 'parameters'],axis=1)
# print(df.columns)
# df['n_vars'] = df['input_stats']['n_vars']

import seaborn as sns

sns.scatterplot(data=df, 
    x='inc', y='idef',hue='gamma',style='method',
    alpha=0.5,
    s=50)
# 



#%%
def select(xaxis, yaxis, **conditions):
    """ e.g.,  
     - select( 'inc', 'idef', method='cvx_opt_joint') 
     - select( 'inc', 'times', method='cvx_opt_joint') 
    """


