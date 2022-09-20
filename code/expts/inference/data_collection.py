"""
Given a PDG, collet 
"""

from collections import namedtuple
import numpy as np

from pgmpy.inference import BeliefPropagation
from pgmpy.utils import get_example_model

import sys
#sys.path.append("../../..")
sys.path.append("../..")
print(sys.path)

from pdg.pdg import PDG
from pdg.store import TensorLibrary
from pdg.rv import Variable as Var
from pdg.dist import CPT, RawJointDist as RJD 

from pdg.alg import interior_pt as ip
from pdg.alg import torch_opt


## TIMING / LOGGING UTILS
import psutil
from psutil._common import bytes2human
# from multiprocessing import Process
import multiprocessing as multiproc
import time
import pickle
import logging


# logging.basicConfig(format='%(asctime)s %(message)s', 
# 	filename='example.log', encoding='utf-8', level=logging.DEBUG)



def wrap(fn, return_bin, fname):
	def fn_wrapped(*args, **kwargs):
		init_time = time.time()
		rslt = fn(*args, **kwargs)
		total_time = time.time() - init_time

		with open(fname+'.pickle', 'wb') as f:
			pickle.dump(rslt, f)

		return_bin.send({'time' : total_time})
		return_bin.send(rslt)
	
	return fn_wrapped


Rslt = namedtuple('computed', ['result', 'total_time', 'max_mem'])
def glog(fname, func, *args, **kwargs):
	recver, sender = multiproc.Pipe(False) #

	p = multiproc.Process(target=wrap(func, sender, fname), args=args, kwargs=kwargs)
	psu_p = psutil.Process(p.pid)

	max_mem = 0
	sleep_time = 1E-4

	p.start()
	sender.close()

	while not recver.poll():
		psu_p = psutil.Process(p.pid)

		max_mem = max(max_mem, psu_p.memory_info().vms)
		time.sleep(sleep_time)
		sleep_time *= 1.5
		# print(bytes2human(psu_p.memory_info().vms))
		print({k : bytes2human(b) for k,b in psu_p.memory_info()._asdict().items()})


	total_time = recver.recv()['time']
	rslt = recver.recv()

	return Rslt(rslt, total_time, max_mem)


def collect_inference_data_for(idstr: str, M:PDG,  store:TensorLibrary=None):
	"""
	run a bunch of algorithms to do inference on the given PDG, 
	write them to a file, and return a tensor library of results.
	"""
	
	if store == None:
		store = TensorLibrary()


	stats = dict(
		n_vars = len(M.varlist),
		n_worlds = np.prod(M.dshape),
		n_params = sum(p.size for p in M.edges('P')),
		n_edges = len(M.Ed)
	)
    
	print(f'{" "+idstr+" ":=^50}')
	print(stats)
	print(f'{"":=^50}')
		

	# for each optimization, log:
	#  - time taken
	#  - memory taken
	#  - training curve (if available): loss over time
	#  - (Inc, Idef) of final product


	def log(idstr, method, *args, **kwargs):
		print('>> ', idstr, method.__name__, args, kwargs)
		dist, total_time, max_mem = glog(idstr, method, M, *args, **kwargs)
		inc = M.Inc(dist).real
		idef = M.IDef(dist)

		print(f'{idstr:<20} \t ',args, kwargs,' \n inc : ', inc,'\t idef: ', idef)

		store(*args,inc=inc,idef=idef, total_time=total_time, max_mem=max_mem,
			**stats, **kwargs).set(dist)

	log(idstr+".ip.-idef", ip.cvx_opt_joint, also_idef=False)
	log(idstr+".ip.+idef", ip.cvx_opt_joint, also_idef=True)

	for gamma in [1E-12, 1E-8, 1E-4, 1E-2, 1]:
		for ozrname in ['adam', "lbfgs", "asgd"]:
			log(idstr+".torch.gamma%.0e"%gamma, 
				torch_opt.opt_dist,
				gamma=gamma, optimizer=ozrname)

		
	## 3 ##  --- torch optimization (Adam)
	## 4 ##  --- torch optimization (LBFGS)
	## 5 ##  --- interior point cvx optimization
	## 6 ##  --- LIR


def colect_data(id:str, bn, store):

	pass

#%%
# %cd ../..
# %pwd
# %load_ext autoreloatd
# %autoreload 2




example_bn_names =  [ 
	"asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
	# "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts", "andes", "diabetes",
	# "link", "munin1", "munin2", "munin3", "munin4", "pathfinder", "pigs", "munin" 
]


if __name__ == '__main__':
	store = TensorLibrary()

	for bn_name in example_bn_names:
		bn = get_example_model(bn_name)
		bp = BeliefPropagation(bn)
		glog(bn_name+"-as-FG.bp", bp.calibrate)


		pdg = PDG.from_BN(bn)
		collect_inference_data_for(bn_name+"-as-pdg", pdg, store)


