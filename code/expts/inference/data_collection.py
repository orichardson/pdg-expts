"""
Given a PDG, collet 
"""

import logging



from pgmpy.inference import ExactInference

from pdg.pdg import PDG
from pdg.store import TensorLibrary
from pdg.rv import Variable as Var
from pdg.dist import CPT, RawJointDist as RJD 


from pdg.alg import interior_pt as ip
from pdg.alg import torch_opt



example_bn_names =  [ 
	"asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
    "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts", "andes", "diabetes",
	"link", "munin1", "munin2", "munin3", "munin4", "pathfinder", "pigs", "munin" ]



# import logging
# logging.basicConfig(format='%(asctime)s %(message)s', 
# 	filename='example.log', encoding='utf-8', level=logging.DEBUG)


def collect_data(idstr:str,  M:PDG,  store:TensorLibrary=None):
	"""
	run a bunch of algorithms to do inference on the given PDG, and
	store the results in the given tensorlibrary (if given).
	
	Also, write them to a file
	Returns a tensor library of results.
	"""
	
	if store == None:
		store = TensorLibrary()
		

	with open(idstr + ".log", 'w') as f:

		mu1 = ip.cvx_opt_clusters(M, also_idef=False)
		mu2 = ip.cvx_opt_clusters(M, also_idef=True)

		f.write()
		f.flush()
		
	## 1 ##  --- 
	## 2 ##  --- 
	## 3 ##  --- torch optimization (Adam)
	## 4 ##  --- torch optimization (LBFGS)
	## 5 ##  --- interior point cvx optimization
	## 6 ##  --- LIR


def colect_data(id:str, bn, store):

	pass

#%%
# %cd ../..
# %pwd
# %load_ext autoreload
# %autoreload 2





#%%



if __name__ == '__main__':
	from pgmpy.readwrite import BIFReader

	reader = BIFReader("alarm.bif")


# from pdg.lib.smoking import M 
# M.optimize_score()