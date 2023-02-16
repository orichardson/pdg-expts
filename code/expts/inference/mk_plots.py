
#%%##########################################
#  IMPORTS AND DATA LOADING
#############################################

import sys; sys.path.append("../..")

import json
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns


# fname = 'tmp-aggregated.json'
fnames = [
### RANDOM GRAPHS FIXED TW
	# ('tw-data-aggregated-1.json', 'tw1'),  # issues: max_tw is wrong;
	# 									# IDef is missing; no optimization baselines.
	# ('tw-aggregated.json', 'tw-all'),
	# ('tw-aggregated-6.json', 'tw6'),
	# ('tw-aggregated-7.json', 'tw7'),
	# ('tw-temp-aggregated.json', 'tw-temp'),
	('cluster_rand_n=1.json', 'tw-test')
### RANDOM GRAPHS JOINT
	# ('random-joint-bad-aggregated.json', 'bad-rand1')
	# ('random-pdg-data-aggregated-6.json', 'rand6'), # 448
	# ('random-pdg-data-aggregated-5.json', 'rand5'), # 7273
	# ('random-pdg-data-aggregated-4.json', 'rand4'),
	# ('random-pdg-data-aggregated-3.json', 'rand3'),
	# ('random-pdg-data-aggregated-2.json', 'rand2'),
	# ('rj-temp-aggregated.json', 'rj-temp'),
### BNS
	# ('datapts-all.json', 'bns0'),  #BNs
	# ('bn-data-aggregated-2.json', 'bns2')
	# ('random-expts-1.csv', 'rand1')
]

dfs = []

for (fname, shortid) in fnames:
	if fname.endswith('.json'):
		with open(fname, 'r') as f:
			tempdf = pd.DataFrame( json.load(f) )

		to_inline = [c for c in ['input_stats', 'parameters', 'rslt_metrics'] if c in tempdf.columns]
		
		dflist = [ tempdf.drop(to_inline, axis=1, inplace=False) ]
		for c in to_inline:
			# special case of "parameters: just want the second component "
			if c == 'parameters': dflist.append(
					pd.json_normalize(pd.json_normalize(tempdf['parameters'])[1]))
			else:
				dflist.append(pd.json_normalize(tempdf[c]))
		tempdf = pd.concat(dflist,axis=1)
		# tempdf = pd.concat([
		# 	tempdf.drop(['input_stats', 'parameters'], axis=1, inplace=False),
		# 	pd.json_normalize(tempdf['input_stats']),
		# 	pd.json_normalize(pd.json_normalize(tempdf['parameters'])[1]) # only kw parameters matter
		# ], axis=1) 

	elif fname.endswith(".csv"):
		tempdf = pd.read_csv(fname)


	if 'graph_id' not in tempdf.columns:
		if not tempdf['input_name'].map(lambda n : '-' in n).any():
			tempdf['graph_id'] = tempdf['input_name']
		else:			
			tempdf['graph_id'] = tempdf['input_name'].apply(lambda inpn: inpn[:inpn.find('-')])
			mapping = tempdf.groupby(['n_worlds', 'n_params', 'n_edges'])['graph_id'].apply(set)
			tempdf.loc[tempdf.method=='cvx_opt_joint', 'graph_id'] = tempdf[
							tempdf.method=='cvx_opt_joint'].apply(
					lambda x: next(iter(n for n in mapping[
						(x.n_worlds, x.n_params, x.n_edges)] if n.isnumeric())), axis=1)

	if len(fnames) > 1:
		tempdf['expt_src'] = fname
		tempdf['old_graph_id'] = tempdf['graph_id']
		tempdf['graph_id'] = tempdf.graph_id.map(str) + shortid

	tempdf = tempdf.loc[:,~tempdf.columns.duplicated()].copy() # get rid of extra "gamma" column
	dfs.append( tempdf )
	print(shortid, '--> ', tempdf.columns)

df0 = pd.concat(dfs, axis='index')
df0.reset_index(inplace=True)
df = df0

#%##########################################
#  DATA ALTERATIONS AND PREPROCESSING
#############################################

df.loc[df.method=='factor_product', 'gamma'] = 1.0

# set up fine method
df['method_fine'] = df['method'].copy()
if 'also_idef' in df.columns:
	cvx_rows = df.method.isin(['cvx_opt_joint','cvx_opt_clusters'])
	df.loc[cvx_rows,'method_fine'] = df[cvx_rows]['also_idef'].map(
			{True : 'cvx+idef', False:'cvx-idef'})

# cccp_rows = df.method.isin(['cccp_opt_joint','cccp_opt_clusters'])
# df.loc[cccp_rows, 'method_fine'] = (df[cccp_rows]['gamma'] <= 1).map(
# 		{True : 'cccp-VEX', False:'cccp-CAVE'})
if 'optimizer' in df.columns:
	torch_rows = df.method.isin(['opt_dist', 'opt_joint','opt_clustree'])
	desc = df[torch_rows].method.map(
		{'opt_dist':'joint', 'opt_joint':'joint', 'opt_clustree':'ctree'})
	df.loc[torch_rows,
		'method_fine'] = 'torch:'+desc+"."+df[torch_rows]['optimizer']

# make sure inc is nonegative (no rounding errors)
df.inc.clip(lower=0, inplace=True)

# calculate objective
# df['obj'] = df['inc'] + df['gamma'] * df['idef']
df['obj'] = df['inc'] + df['gamma'] * df['idef'] * np.log(2)

# calculate memdif
if 'init_mem' in df.columns:
	df['mem_diff'] = df.max_mem - df.init_mem

# calculate gap
best = {}
winner = {}
for gid,gamma in df[['graph_id', 'gamma']].value_counts().index:
	samegraph = df[(df.graph_id == gid)]
	# best[(gid,gamma)] = (samegraph.inc + gamma*samegraph.idef).min()
	scores = (samegraph.inc + gamma*samegraph.idef * np.log(2))
	best[(gid,gamma)] = scores.min()
	winner[(gid,gamma)] = samegraph.iloc[scores.argmin()]['index']
df['gap'] = df.apply(lambda x: x.obj - best[(x.graph_id, x.gamma)], axis=1)

# Lower bounds for log plots
MIN = 1E-15
df[['gap','gamma']] += MIN 
df['noisy_gap'] = df.gap + MIN * np.random.rand(len(df))*4


df['n_params_smoothed'] = df.n_params.map(lambda x: round(x,-2))
if 'n_VC' in df.columns:
	df['n_VC_smoothed'] = df.n_VC.map(lambda x: round(x,-2))
if 'n_worlds' in df.columns:
	df['n_worlds_smoothed'] = df.n_worlds.map(lambda x: round(x,-2))

# df_lr_curated = df.copy()
# if 'lr' in df.columns:
ozr_axes = {'lr','optimizer', 'representation'} & set(df.columns)
to_select = {'lr'}
df_best_hyperparam_idx = df.sort_values('gap').groupby(
		['gamma', 'graph_id', 'method'] + list(ozr_axes - to_select) ).head(1).index
df_hyper_curated = df[(~torch_rows) | df.index.isin(df_best_hyperparam_idx)]


#############################
# set up colors
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

mpalette = {
	'torch:ctree.adam' : 'tab:green',
	'torch:joint.adam' : 'tab:green',
	'torch:ctree.lbfgs' : 'darkgreen',
	'torch:joint.lbfgs': 'darkgreen',
	'factor_product': 'slategray',
    'cccp_opt_joint': 'purple',
	'cccp_opt_clusters': 'purple', 
    'cvx+idef': 'goldenrod',
    'cvx-idef': 'darkgoldenrod',
}
morder = list(reversed(mpalette.keys()))
df = df.iloc[df.method_fine.map(lambda mf : -morder.index(mf)).sort_values().index]

#%%##########################################
#  PLOTTING FUNCTIONS
#############################################
# sns.set_theme()

def plot_grid(data, x_attrs, y_attrs, plotter, condition=None, **kws):
	if condition:
		data = data[condition]

	fig, AX = plt.subplots(len(y_attrs), len(x_attrs), figsize=(15,12), squeeze=False)

	for i,xa in enumerate(x_attrs):
		for j,ya in enumerate(y_attrs):
			plotter(data=data, x=xa, y=ya, ax=AX[j,i], **kws)

	plt.show()


#%%##########################################
#############################################
########  ONE-OFF PLOTTING CELLS  ###########
#############################################
#############################################


#%%###########################################
###   1.   Resources vs Problem Size       ###
##############################################
# df1 = df[df.gamma >= 1E-9]
df1=df_hyper_curated
wmeasure = 'n_VC' if 'n_VC_smoothed' in df1.columns else 'n_worlds'

fig, AX = plt.subplots(2, 2, figsize=(15,15))
sns.lineplot(data=df1,
	x='n_params_smoothed', y='total_time', hue='method_fine', ax=AX[0][0],palette=mpalette)
sns.lineplot(data=df1,
	x=wmeasure, y='total_time', hue='method_fine', ax=AX[0][1],palette=mpalette)
sns.lineplot(data=df1,
	x='n_params_smoothed', y='max_mem', hue='method_fine', ax=AX[1][0],palette=mpalette)
sns.lineplot(data=df1,
	x=wmeasure, y='max_mem', hue='method_fine', ax=AX[1][1],palette=mpalette)



#%% #########################################
#####       2.    Time vs Gap           #####
#############################################
df1 = df_hyper_curated
df1 = df1[df1.method != 'factor_product']
# df1['noisy_gap'] = df1.gap + MIN * np.random.rand(len(df1))
fig, AX = plt.subplots(1, 1, figsize=(10,5))
AX.set(yscale='log', 
	xscale='log'
	)
sns.scatterplot(data=df1,
	# x="gap",
	x="noisy_gap",
	y="total_time", 
	hue="method_fine",ax=AX,
	s=80,
	# s = 15 + df1.n_worlds/20,
	# s=15 + df1.n_VC/10,
	alpha=0.4,
	linewidth=0,
	palette=mpalette,
	# hue_order=morder
	)
sns.despine()
AX.set_ylabel("total time (s)")
		# ax.set_xlabel("objective value")
AX.set_xlabel("gap between objective and best known objective for this graph (plus %.0e floor)"%MIN)
AX.legend(title="Algorithm")
fig.tight_layout()

#%% #############################################
####   2a. Time vs Obj(or Gap), by Gamma    #####
#################################################
# df1 = df[df.gamma >= 1E-9]
df1 = df
gammas = sorted(df1.gamma.unique())
methods_fine = df1.method_fine.unique()
fig, AX = plt.subplots(2, len(gammas), figsize=(18,12), sharey=True)
for i,yattr in enumerate(['obj', 'gap']):
	for j,(ax,g) in enumerate(zip(AX[i],gammas)):
		# ax.set(yscale='log', xscale='log')
		ax.set(yscale='log')
		if yattr == 'gap':
			ax.set(xscale='log')
		ax.set_title("($\gamma = 10^{%d}$)"%(int(round(np.log10(g)))))

		dfg = df1[df1.gamma==g]
		sns.scatterplot(data=dfg, x=yattr, y="total_time", 
			hue="method_fine",
			# style="expt_src",
			# hue_order=methods_fine,
			# s=15 + dfg.n_VC/10,
			# s=15 + dfg.n_worlds/20,
			# s=100,
			s=80,
			alpha=0.5,
			palette=mpalette,
			ax=ax)
		
		ax.set_ylabel("total time (s)")
		# ax.set_xlabel("objective value")
		sns.despine()

		ax.set_xlabel(yattr)
		if i > 0 or j > 0:
			# pass
			# ax.legend.remove()
			ax.legend().set_visible(False)
fig.tight_layout()

#%% #############################################
####   3.    gap vs problem size            #####
#################################################
df1 = df_hyper_curated
wmeasure = 'n_VC' if 'n_VC_smoothed' in df1.columns else 'n_worlds_smoothed'
# df1['noisy_gap'] = df1.gap + MIN * np.random.rand(len(df1))
fig, AX = plt.subplots(1, 1, figsize=(10,10))
AX.set(yscale='log', 
	xscale='log'
	)
sns.lineplot(data=df1,
	# x="gap",
	# x="noisy_gap",
	# y="total_time", 
	x=wmeasure,
	# y='noisy_gap',
	y='gap',
	hue="method_fine",ax=AX,
	# s=80,
	# s = 15 + df1.n_worlds/20,
	# s=15 + df1.n_VC/10,
	# alpha=0.25,
	# linewidth=0,
	palette=mpalette
	)
#%% #############################################
####   4.            #####
#################################################
dfsmall = df
# dfsmall = df[df.gamma >= 1E-9]

fig, AX = plt.subplots(1, 1, figsize=(10,10))
# AX.set(xscale='log',yscale='log')
# sns.scatterplot(data=dfsmall, 
#     x=df.gamma * (np.random.rand(len(df))/2+0.6), y='gap', hue='method', 
#     linewidth=0, alpha=0.4, s=50,
#     ax=AX)
# AX.set(xscale='log')
sns.stripplot(data=dfsmall, 
	x= np.round(np.log10(df.gamma),1).astype(str),
	y="gap",
	hue='method_fine', 
	# order=sorted(np.round(np.log10(df.gamma),1).astype(str).unique(),key=float),
	# s=10 + np.log(dfsmall.n_worlds)/1,
	# s= 2 + dfsmall.n_worlds / 500,
	# s = 2 + dfsmall.n_VC / 200,
	s=8,
	# linewidth=np.log(dfsmall.n_worlds)/1,
	linewidth=1,
	alpha=0.1,
	ax=AX)



#%% ##############################################
####  5. violin plot: gap by method_fine     #####
##################################################

sns.violinplot(data=df, x='method', y='total_time')


#%% ############################################
df1 = df

fig, AX = plt.subplots(2, 2, figsize=(20,20))
AX[0,0].set(yscale='log')
sns.scatterplot(data=df1, 
    x=df.n_VC * (np.random.rand(len(df))/4+0.6), y='gap', hue='method_fine', 
    linewidth=0, alpha=0.4, s=50,
    ax=AX[0,0])
# AX.set(xscale='log')
# sns.stripplot(data=df1, 
# 	# x= np.round(np.log10(df1.gamma),1).astype(str),
# 	x=df1.graph_id.astype(str),
# 	y="gap",
# 	hue='method_fine', 
# 	# order=sorted(np.round(np.log10(df.gamma),1).astype(str).unique(),key=float),
# 	# s=10 + np.log(dfsmall.n_worlds)/1,
# 	# s= 2 + dfsmall.n_worlds / 500,
# 	# s = 2 + dfsmall.n_VC / 200,
# 	s=8,
# 	# linewidth=np.log(dfsmall.n_worlds)/1,
# 	linewidth=1,
# 	alpha=0.1,
# 	ax=AX)



#%%
# ax = plt.
sns.scatterplot(data=df, x=df.iters, y=np.log(df.gap), hue='method_fine')


#%%
# plot_grid(df, 
# 	# ['n_params','n_worlds', 'max_tw', 'n_vars'],	
# 	['n_params','n_worlds', 'representation', 'iters', 'n_vars'],	
# 	# ['mem_diff', 'total_time',  'inc'], 
# 	['gap', 'mem_diff', 'total_time', 'obj', 'inc'], 
# 	sns.lineplot,
# 	hue='method_fine')

#%%

sns.scatterplot(data=df, 
	x='total_time', y='obj',hue='method',
	# hue_order=['opt_dist, 'cccp_opt_joint', 'cvx_opt_joint'],
	hue_order=['opt_clustree', 'cccp_opt_clusters', 'cvx_opt_clusters'],
	alpha=0.2, # cmap=blu_org,
	s=50,linewidth=0)


#%% ####
# sns.scatterplot(data=df, 
#     x='total_time', y='obj',hue='method',style='method',
#     hue_order=['opt_dist', 'cccp_opt_joint', 'cvx_opt_joint'],
#     alpha=0.2, # cmap=blu_org,
#     s=50,linewidth=0)
# df1 = df[df.gamma <= 1]
df1 = df
fig, AX = plt.subplots(2, 2, figsize=(15,15))
sns.lineplot(data=df1,
	x='n_VC', y='total_time', hue='method_fine', ax=AX[0][0])
sns.lineplot(data=df1,
	x='n_worlds', y='total_time', hue='method_fine', ax=AX[0][1])
sns.lineplot(data=df1,
	x='n_params', y='max_mem', hue='method_fine', ax=AX[1][0])
sns.lineplot(data=df1,
	x='n_VC', y='max_mem', hue='method_fine', ax=AX[1][1])







#%% 
df1 = df
# df1 = df[df.gamma <= 1]
fig, AX = plt.subplots(2, 2, figsize=(20,20))
sns.lineplot(data=df1,
	x='n_params', y='gap', hue='method_fine', ax=AX[0,0])
sns.lineplot(data=df1,
	# x='n_worlds', y='gap', hue='method_fine', ax=AX[0,1])
	x='n_VC', y='gap', hue='method_fine', ax=AX[0,1])

# sns.kdeplot(data=df,
#     x='n_params', y='gap', hue='method', ax=AX[1,0])
# sns.kdeplot(data=df,
#     x='n_worlds', y='gap', hue='method', ax=AX[1,1])

# AX[1,1].set(yscale='log')
# sns.scatterplot(data=df[df.method==''],
#     x='n_worlds', y='gap', hue='method', ax=AX[1,1])

AX[1,0].set(yscale='log')
sns.stripplot(data=df1, # s=100, linewidth=0,alpha=0.4,
		# alpha=0.4, order=['opt_dist', 'cccp_opt_joint', 'cvx_opt_joint'],
		alpha=0.4,
	x='n_worlds', y='gap', hue='method_fine', ax=AX[1,0])
AX[1,1].set(yscale='log', xscale='log')
sns.lineplot(data=df1,
	x='gamma', y='gap', hue='method_fine', ax=AX[1,1])
# AX[2,1].set(xscale='log')
# sns.swarmplot(data=df,
#     x='gap', hue='gamma', y='method', hue_norm=LogNorm(), ax=AX[2,1])

#%% # Something interesing here... 
###############################################
# What about if we zoom in on # worlds < 2k?
# dfsmall = df[df.n_worlds < 3000]
dfsmall = df
# dfsmall = df[df.gamma >= 1E-9]

fig, AX = plt.subplots(1, 1, figsize=(10,10))
# AX.set(xscale='log',yscale='log')
# sns.scatterplot(data=dfsmall, 
#     x=df.gamma * (np.random.rand(len(df))/2+0.6), y='gap', hue='method', 
#     linewidth=0, alpha=0.4, s=50,
#     ax=AX)
# AX.set(xscale='log')
sns.stripplot(data=dfsmall, 
	x= np.round(np.log10(df.gamma),1).astype(str),
	y="gap",
	hue='method_fine', 
	# order=sorted(np.round(np.log10(df.gamma),1).astype(str).unique(),key=float),
	# s=10 + np.log(dfsmall.n_worlds)/1,
	# s= 2 + dfsmall.n_worlds / 500,
	# s = 2 + dfsmall.n_VC / 200,
	s=8,
	# linewidth=np.log(dfsmall.n_worlds)/1,
	linewidth=1,
	alpha=0.1,
	ax=AX)


#%%
df1 = df
# fig, AX = plt.subplots(3, 2, figsize=(15,12))

# sns.pairplot(data=df['n_worlds', 'total_time', 'mem_diff'])

#%% ####
# df1 = df[df.n_worlds <= 4000]
df1 = df
fig, AX = plt.subplots(2, 2, figsize=(15,15))
AX[0,0].set(yscale='log')
sns.lineplot(data=df1,
	x='n_worlds', y='gap', hue='method', ax=AX[0][0])
sns.lineplot(data=df1,
	x='n_worlds', y='total_time', hue='method', ax=AX[0][1])
AX[1,0].set(yscale='log')
sns.lineplot(data=df1,
	x='n_worlds', y='gap', hue='method', ax=AX[1][0])
sns.lineplot(data=df1,
	x='n_worlds', y='max_mem', hue='method', ax=AX[1][1])



#%% 

fig, AX = plt.subplots(2, 2, figsize=(20,16))
AX[0,0].set(xscale='log')
sns.stripplot(data=df, x='gap', y='method', hue='total_time', ax=AX[0,0])



#%%%

# ax = sns.scatterplot(data=df,
#     x='n_worlds', y='gap', hue='method')
# ax.set(yscale='log')

sns.kdeplot(data=df[df.method=='cccp_opt_joint'],
	x=np.log(df[df.method=='cccp_opt_joint'].gamma), y='obj', fill=True)


#%%

#%%
# Just plot inc vs idef, colored by gamma. Looks cool.

fig, ax = plt.subplots(1,1)
ax.set(xscale='log')
sns.scatterplot(data=df, 
	# x='inc',
	x=df.inc + 1E-8,
	y='idef',hue='gamma',hue_norm=LogNorm(),style='method',
	alpha=0.5, s=50, linewidth=0,
	ax=ax)


#%%%
## Same as above (inc vs idef, only looks cool), but with log scale
fig, ax = plt.subplots(1,1)
ax.set(xscale='log')
sns.scatterplot(data=df, 
	# x='inc',
	x=df.inc + 1E-20,
	y='idef',hue='method_fine',style='method_fine',
	alpha=0.5, s=50, linewidth=0,
	ax=ax)



#%%
#
fig, ax = plt.subplots(1,1)
ax.set(xscale='log')
sns.stripplot(data=df, 
	x='total_time', y='method', hue='gap',
	alpha=0.5, linewidth=0,
	hue_norm=LogNorm(),
	ax=ax
	)




#%%  
blu_org = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

ax = sns.scatterplot(data=df, 
	x='inc', y='idef',hue='method',style='method',
	hue_order=['opt_dist', 'cccp_opt_joint', 'cvx_opt_joint'],
	alpha=0.2,  cmap=blu_org,
	s=50,linewidth=0)

idef_range = np.linspace(df['idef'].min(), df['idef'].max(), 100)
pareto_opt = [df[df['idef'] <= idf]['inc'].min() for idf in idef_range]

sns.lineplot(y=idef_range, x=pareto_opt, orient='y', ax=ax)


sns.pairplot(data=df[['inc', 'idef', 'total_time', 'gamma', 'method']], hue='method',
	plot_kws=dict(alpha=0.5))

#%%
sns.displot(data=df, x='inc', col='method', kde=True)

#%% 
sns.kdeplot(data=df, x='inc', y='total_time', hue='method', fill=True, alpha=0.5, levels=100)

#%%
sns.kdeplot(data=df, x='inc', y='idef', hue='method', fill=True, alpha=0.9, levels=10)

#%%
sns.set_theme() 
sns.histplot(data=df, x='inc', y='total_time', hue='method', fill=True,  bins=50, pthresh=.1, cmap="mako")

#%%
# sns.jointplot(data=df, x='inc', y='total_time', kind='hex')
sns.jointplot(data=df, x='n_worlds', y=np.log(df.gap), kind='hex')

#%%
# sns.set_style("ticks")
sns.violinplot(data=df, x='method', y='total_time')
# sns.despine(offset=10, trim=True);

