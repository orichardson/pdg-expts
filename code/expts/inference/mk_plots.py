
#%%
import sys; sys.path.append("../..")

from matplotlib import pyplot as plt
import numpy as np
import json

fname = 'random-pdg-data-aggregated-3.json'
# fname = 'random-pdg-data-aggregated-2.json'
# fname ='datapts-all.json'
with open(fname, 'r') as f:
    data = json.load(f)

#%%
# from pdg.store import TensorLibrary
# lib = TensorLibrary()

pt = data[0]
pt
# del pt['input_stats']
# lib(**pt)

# for pt in data:

import pandas as pd
df = pd.DataFrame(data)



## maybe drop dicts?
# df.drop(['input_stats', 'parameters'],axis=1)
## maybe drop calibrate?
# df.drop(index=df[(df['method'] =='calibrate')].index,inplace=True)

df = pd.concat([
    df.drop(['input_stats', 'parameters'], axis=1, inplace=False),
    pd.json_normalize(df['input_stats']),
    pd.json_normalize(pd.json_normalize(df['parameters'])[1]) # only kw parameters matter
], axis=1) 
# df = pd.json_normalize(df)

df = df.loc[:,~df.columns.duplicated()].copy() # get rid of extra "gamma" column

if 'graph_id' not in df.columns:
    df['graph_id'] = df['input_name'].apply(lambda inpn: inpn[:inpn.find('-')])

    mapping = df.groupby(['n_worlds', 'n_params', 'n_edges'])['graph_id'].apply(set)

    df.loc[df.method=='cvx_opt_joint', 'graph_id'] = df[df.method=='cvx_opt_joint'].apply(
            lambda x: next(iter(n for n in mapping[
            (x.n_worlds, x.n_params, x.n_edges)] if n.isnumeric())), axis=1)

# df.loc[df.method=='cvx_opt_joint', 'graph_id'] = None
# df['inc_gap'] = df['graph_id']

df['gamma'] += 1E-15


#%%
from matplotlib.colors import LogNorm
import seaborn as sns

sns.scatterplot(data=df, 
    x='inc', y='idef',hue='gamma',hue_norm=LogNorm(),style='method',
    alpha=0.5,
    s=50)

#%%

sns.swarmplot(data=df, 
    x='total_time', y='method', hue='inc',
    # hue_norm=LogNorm()
    )

#%%
sns.lineplot(data=df,x='total_time', hue='method', y='inc')



#%%
blu_org = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

ax = sns.scatterplot(data=df, 
    x='inc', y='idef',hue='method',style='method',
    hue_order=['opt_dist', 'cccp_opt_joint', 'cvx_opt_joint'],
    alpha=0.2, # cmap=blu_org,
    s=50,linewidth=0)

idef_range = np.linspace(df['idef'].min(), df['idef'].max(), 100)
pareto_opt = [df[df['idef'] <= idf]['inc'].min() for idf in idef_range]

sns.lineplot(y=idef_range, x=pareto_opt, orient='y', ax=ax)




#%%
sns.set_theme()
sns.pairplot(data=df[['inc', 'idef', 'total_time', 'gamma', 'method']], hue='method',
    plot_kws=dict(alpha=0.5))

#%%
sns.displot(data=df, x='inc', col='method', kde=True)

#%% 
sns.kdeplot(data=df, x='inc', y='total_time', hue='method', fill=True, alpha=0.5, levels=100)

#%%
sns.kdeplot(data=df, x='inc', y='idef', fill=True, alpha=0.9, levels=10)

#%%
sns.set_theme() 
sns.histplot(data=df, x='inc', y='total_time', hue='method', fill=True,  bins=50, pthresh=.1, cmap="mako")

#%%
sns.jointplot(data=df, x='inc', y='total_time', kind='hex')

#%%
# sns.set_style("ticks")
sns.violinplot(data=df, x='method', y='total_time')
# sns.despine(offset=10, trim=True);


#%%
def select(xaxis, yaxis, **conditions):
    """ e.g.,  
     - select( 'inc', 'idef', method='cvx_opt_joint') 
     - select( 'inc', 'times', method='cvx_opt_joint') 
    """


