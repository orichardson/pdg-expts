import argparse
parser = argparse.ArgumentParser(description="collect experimental data from Bayesian Networks")
parser.add_argument("--data-dir", dest='datadir', type=str, 
	default='random-pdg-data',
	help="the name of directory to store points in")

parser.add_argument("-N", "--num-pdgs", default=1000, type=int,
	help="number of PDGs to generate.")
parser.add_argument("-n", "--num-vars", nargs= 2, default=[9,10], type=int,
	help="number of variables ine each PDG")
parser.add_argument("-e", "--edge-range", nargs=2, default=[8,15], type=int,
	help="number of pdg edges to generate (upper & lower).")
parser.add_argument( "--num-edges", type=int,help="number of PDGs to generate.")
parser.add_argument("-v", "--num-vals", nargs=2, type=int,
	default=[2,2],
	help="range of values (upper & lower) for each variable")
parser.add_argument("-s", "--src-range", nargs=2, type=int,
	default=[0,3],
	help="bounds for how many sources each edge can have")
parser.add_argument("-t", "--tgt-range", nargs=2,  type=int,
	default=[1,2],
	help="bounds for how many targets each edge can have")
parser.add_argument("-z", "--ozrs", nargs='*', type=str,
	default=['adam', 'lbfgs' ,'asgd'],
	help="Which optimizers to use? Choose from {adam, lbfgs, asgd, sgd}")
parser.add_argument("-r", "--reprs", nargs='*', type=str,
	default=['simplex', 'gibbs'],
	help="Which representation to use? Choose from {simplex, gibbs, soft-simplex}")
# parser.add_argument("-i", "--num-iters", nargs='*', type=int,
# 	default=20,
# 	help="Maximum Iters")

parser.add_argument("-l", "--learning-rates", nargs='*', type=float,
	# default=[1E0, 1E-1, 1E-2, 1E-3, 1E-4],
	default=[],
	help="Learning Rates")

parser.add_argument("-g", "--gammas", nargs='*', type=float,
	default=[1E-12, 1E-8, 1E-4, 1E-2, 1, 2],
	help="Selection of gamma values")
parser.add_argument("--num-cores", type=int, default=-1)
parser.add_argument("--verbose", action="store_true", default=False)

args=parser.parse_args()


import numpy as np
import networkx as nx
import json
import random
from functools import reduce
# from itertools import chain
from operator import and_

# from pgmpy.inference import BeliefPropagation
import sys; sys.path.append("../..")

from pdg.pdg import PDG
from pdg.rv import Unit, Variable as Var
from pdg.dist import CPT, RawJointDist as RJD

from pdg.alg import interior_pt as ip
from pdg.alg import torch_opt

import os
import signal
import pickle
from expt_utils import MultiExptInfrastructure


global expt
expt = MultiExptInfrastructure(args.datadir, n_threads=args.num_cores)

def terminate_signal(signalnum, *args):
	global expt
	expt.finish_now = True

signal.signal(signal.SIGINT, terminate_signal)
signal.signal(signal.SIGTERM, terminate_signal)


import itertools as itt
def reset():
	global var_names
	var_names = iter(itt.chain(
		(chr(i + ord('A')) for i in range(26)) ,
		("X%d_"%v for v in itt.count()) ))
	
reset()

verb = args.verbose

try:
	for i in range(args.num_pdgs):
		if expt.finish_now:
			print("Exiting!")
			break

		reset(); global var_names


		pdg = PDG()
		n = random.randint(*args.num_vars)
		for _ in range(n):
			pdg += Var.alph(next(var_names), random.randint(*args.num_vals))

		num_edges = args.num_edges if args.num_edges else random.randint(*args.edge_range)
		print(args, 'num_edges' in args, num_edges)
		for e in range(num_edges):
			src = random.sample(pdg.varlist, k=random.randint(*args.src_range))
			# print('remaining', [ v for v in pdg.varlist if v not in src])
			# print('args.tgt_range: ', args.tgt_range)
			tgt = random.sample([ v for v in pdg.varlist if v not in src], k=random.randint(*args.tgt_range))

			# print(src, tgt)

			# pdg += CPT.make_random( reduce(and_, src, initial=Unit), reduce(and_, tgt, initial=Unit) )
			print(f"{Var.product(src).name:>20} --> {Var.product(tgt).name:<20}")
			pdg += CPT.make_random( Var.product(src), Var.product(tgt))

		with open(args.datadir+"/%d.pdg" % i, 'wb') as fh:
			pickle.dump(pdg, fh)
			
		stats = dict(
			graph_id = i,
			n_vars = len(pdg.varlist),
			n_worlds = int(np.prod(pdg.dshape)), # without cast, json cannot interperet int64 -.-
			n_params = int(sum(p.size for p in pdg.edges('P'))), #here also	
			n_edges = len(pdg.Ed)
		)
					
		# expt.enqueue(str(i), stats, ip.cvx_opt_joint, pdg, also_idef=False)
		# expt.enqueue(str(i), stats, ip.cvx_opt_joint, pdg, also_idef=True)
		expt.enqueue("%d--cvx-idef"%i, stats, ip.cvx_opt_joint, pdg, also_idef=False)
		expt.enqueue("%d--cvx+idef"%i, stats, ip.cvx_opt_joint, pdg, also_idef=True)
		#,verbose=verb
		# collect_inference_data_for(bn_name+"-as-pdg", pdg, store)

		for gamma in args.gammas:
			expt.enqueue("%d--cccp--gamma%.0e"%(i,gamma), stats,
								ip.cccp_opt_joint, pdg, 
								gamma=gamma) #, verbose=verb
			# expt.enqueue(str(i), stats, ip.cccp_opt_joint, pdg, gamma=gamma)
			
			# for ozrname in ['adam', "lbfgs", "asgd"]:
			for rep in args.reprs:
				for ozrname in args.ozrs:
					# for oi in niters[ozrname]:
					if len(args.learning_rates) == 0:
						expt.enqueue("%d--torch(%s;%s)--gamma%.0e"%
									(i,ozrname,rep,gamma), stats,
							torch_opt.opt_joint, pdg,
							gamma=gamma, optimizer=ozrname,
							representation=rep)

					for lr in args.learning_rates:
						expt.enqueue("%d--torch(%s;%s@%.0f)--gamma%.0e"%
									(i,ozrname,rep,lr,gamma), stats,
							torch_opt.opt_joint, pdg,
							gamma=gamma, optimizer=ozrname, lr=lr,
							representation=rep)


		

		#Finally, just multiply the cpds for gamma = 1. 
		expt.enqueue(
			"%d--factor-multiplication", dict(gamma=1, **stats),
				PDG.factor_product, pdg)
	
	expt.done()
except (KeyboardInterrupt, InterruptedError) as e:
	print("Interrupted! Dumping results ... ")

except Exception as e:
	print("Uh-oh...", e)

finally:
	with open(args.datadir+"/RESULTS.json", 'w') as f:
		json.dump([r._asdict() for r in expt.results.values() if r is not None ], f)
	
	print('... finished writing to "RESULTS.json! ')


	# with open("library.pickle", 'w') as f:
	# 	pickle.dump(store, f)
	# print(expt.results)

