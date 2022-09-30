import argparse
parser = argparse.ArgumentParser(description="collect experimental data from Bayesian Networks")
parser.add_argument("--data-dir", dest='datadir', type=str, 
	default='random-pdg-data',
	help="the name of directory to store points in")

parser.add_argument("-N", "--num-pdgs", default=1000, type=int,
	help="number of PDGs to generate.")
parser.add_argument("-n", "--num-vars", default=10, type=int,
	help="number of PDGs to generate.")
parser.add_argument("-e", "--num-edges", default=12, type=int,
	help="number of PDGs to generate.")
parser.add_argument("-v", "--num-vals", nargs=2, type=int,
	default=[2,2],
	help="range of values (upper & lower) for each variable")
parser.add_argument("-s", "--src-range", nargs=2, 
	default=[0,3],
	help="bounds for how many sources each edge can have")
parser.add_argument("-t", "--tgt-range", nargs=2, 
	default=[1,2],
	help="bounds for how many targets each edge can have")
parser.add_argument("-z", "--ozrs", nargs='*', 
	default=['adam', 'lbfgs' ,'asgd'],
	help="Which optimizers to use? Choose from {adam, lbfgs, asgd, sgd}")
parser.add_argument("-g", "--gammas", nargs='*', 
	default=[1E-12, 1E-8, 1E-4, 1E-2, 1, 2],
	help="Selection of gamma values")


args=parser.parse_args()


import numpy as np
import networkx as nx
import json
import random
from functools import reduce
from operator import and_

# from pgmpy.inference import BeliefPropagation
import sys; sys.path.append("../..")

from pdg.pdg import PDG
from pdg.rv import Unit, Variable as Var
from pdg.dist import CPT, RawJointDist as RJD

from pdg.alg import interior_pt as ip
from pdg.alg import torch_opt


from expt_utils import MultiExptInfrastructure


if __name__ == '__main__':
	expt = MultiExptInfrastructure(args.datadir)

	for i in range(args.num_pdgs):
		pdg = PDG()
		for v in range(args.num_vars):
			pdg += Var.alph("X%d_"%v, random.randint(*args.num_vals))

		for e in range(args.num_edges):
			src = random.sample(pdg.varlist, k=random.randint(*args.src_range))
			tgt = random.sample([ v for v in pdg.varlist if v not in src], k=random.randint(*args.tgt_range))
			# print(src, tgt)

			# pdg += CPT.make_random( reduce(and_, src, initial=Unit), reduce(and_, tgt, initial=Unit) )
			print(f"{Var.product(src).name:>20} --> {Var.product(tgt).name:<20}")
			pdg += CPT.make_random( Var.product(src), Var.product(tgt))
			
		stats = dict(
			n_vars = len(pdg.varlist),
			n_worlds = int(np.prod(pdg.dshape)), # without cast, json cannot interperet int64 -.-
			n_params = int(sum(p.size for p in pdg.edges('P'))), #here also	
			n_edges = len(pdg.Ed)
		)
					
		expt.enqueue("cvx-idef", stats, ip.cvx_opt_joint, pdg, also_idef=False)
		expt.enqueue("cvx+idef", stats, ip.cvx_opt_joint, pdg, also_idef=True)
		# collect_inference_data_for(bn_name+"-as-pdg", pdg, store)

		for gamma in args.gammas:
			expt.enqueue("%d--cccp--gamma%.0e"%(i,gamma), stats,
					 ip.cccp_opt_joint, pdg, gamma=gamma)
			
			# for ozrname in ['adam', "lbfgs", "asgd"]:
			for ozrname in args.ozrs:
				expt.enqueue("%d--torch(%s)--gamma%.0e"%(i,ozrname,gamma), stats,
					torch_opt.opt_dist, pdg,
					gamma=gamma, optimizer=ozrname)
				
	
	expt.done()

	# with open("library.pickle", 'w') as f:
	# 	pickle.dump(store, f)
	with open("RESULTS.json", 'w') as f:
		json.dump(expt.results, f)

