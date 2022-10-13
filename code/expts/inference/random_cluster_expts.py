import argparse
parser = argparse.ArgumentParser(description="collect experimental data from Bayesian Networks")
parser.add_argument("--data-dir", dest='datadir', type=str, 
	default='random-pdg-tw-data',
	help="the name of directory to store points in")

parser.add_argument("-N", "--num-pdgs", default=1000, type=int,
	help="number of PDGs to generate.")
# parser.add_argument("-c", "--num-clusters", nargs= 2, default=[3,6], type=int,
	# help="number of variables ine each PDG")
parser.add_argument("-e", "--edge-range", nargs=2, default=[8,15], type=int,
	help="number of pdg edges to generate (upper & lower).")
parser.add_argument("-n", "--num-vars", nargs=2, default=[8,30], type=int,
	help="number of variables per cluster.")
parser.add_argument("-W", "--tw", default=[1,2], type=int, nargs=2,
	help="treewidth.")
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

parser.add_argument("-i", "--num-iters", nargs='*', type=int,
	default=20,
	help="How many iterations?")

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

def find_cliques_size_k(G, k):
	""" based on https://stackoverflow.com/a/58782120/13480314 """
	for clique in nx.find_cliques(G):
		if len(clique) == k:
			yield tuple(clique)
		elif len(clique) > k:
			yield from itt.combinations(clique,k)

def random_k_tree(n, k):
	if n <= k+1:
		G = nx.complete_graph(n)
		ctree = nx.Graph(); ctree.add_node(tuple(G.nodes()))
		return G, ctree

	G = nx.complete_graph(k + 1)
	ctree = nx.Graph()
	ctree.add_node(tuple(G.nodes()))

	while len(G.nodes()) < n:
		kcq = random.choice( list(find_cliques_size_k(G,k)))
		
		newnode = len(G.nodes())
		newcluster = kcq + (newnode, )

		G.add_node(newnode)
		ctree.add_node(newcluster)

		G.add_edges_from( (n, newnode) for n in kcq )
		ctree.add_edges_from((C, newcluster) for C in ctree.nodes() if len(set(C) & set(newcluster)) == k)

	ctree_tree = nx.minimum_spanning_tree(ctree)
	return G, ctree_tree


def pprocessor(M):
	def process_pseudomarginals(cpm):
		# print(cpm.inc, cpm.idef, cpm.cluster_dist)
		assert np.allclose([cpm.inc, cpm.idef], [M.Inc(cpm.cluster_dist), M.IDef(cpm.cluster_dist)])
		return (cpm.inc, cpm.idef)
	return process_pseudomarginals

try:
	for i in range(args.num_pdgs):
		if expt.finish_now:
			print("Exiting!")
			break

		reset(); global var_names

		pdg = PDG()

		clusters = []
		# for j in range(args.num_clusters):
		# 	clusters.append([
		# 		Var.alph(next(var_names), random.randint(*args.num_vals))
		# 		for _ in range(random.randint(*args.vars_per_cluster))
		# 	])
		# ctree = nx.random_tree(args.num_clusters)

		n = random.randint(*args.num_vars)
		k = random.randint(*args.tw)
		m = random.randint(*args.edge_range)

		g, ctree = random_k_tree(n,k)

		for _, vn in zip(range(n), var_names):
			pdg += Var.alph(vn, random.randint(*args.num_vals))

		while len(pdg.edgedata) < m:
			# c1,c2 = random.choice(list(ctree.edges()))
			c1 = random.choice(list(ctree.nodes()))
			# c2 = random.choice(list(ctree[c1]))
			# options = [pdg.varlist[i] for i in set(c1) | set(c2)]
			options = [pdg.varlist[i] for i in c1]
			# options = random.choice(list(ctree.nodes()))

			try:
				src = random.sample(options, k=random.randint(*args.src_range))
				# print('remaining', [ v for v in pdg.varlist if v not in src])
				# print('args.tgt_range: ', args.tgt_range)
				tgt = random.sample([ v for v in options if v not in src], k=random.randint(*args.tgt_range))
			except ValueError:
				continue # if there wasn't space, try again. 

			# pdg += CPT.make_random( reduce(and_, src, initial=Unit), reduce(and_, tgt, initial=Unit) )
			print(f"{Var.product(src).name:>20} --> {Var.product(tgt).name:<20}")
			pdg += CPT.make_random( Var.product(src), Var.product(tgt))

		nx.relabel_nodes(ctree, {C:tuple(pdg.varlist[i].name for i in C) for C in ctree.nodes()}, copy=False)
		print("CTREE NODES", ctree.nodes())

		with open(args.datadir+"/%d.pdg" % i, 'wb') as fh:
			pickle.dump(pdg, fh)
			
		stats = dict(
			graph_id = i,
			max_tw = args.tw,
			n_vars = len(pdg.varlist),
			n_worlds = int(np.prod(pdg.dshape)), # without cast, json cannot interperet int64 -.-
			n_params = int(sum(p.size for p in pdg.edges('P'))), #here also	
			n_edges = len(pdg.Ed)
		)
					
		# expt.enqueue(str(i), stats, ip.cvx_opt_joint, pdg, also_idef=False)
		# expt.enqueue(str(i), stats, ip.cvx_opt_joint, pdg, also_idef=True)
		# expt.enqueue("%d--cvx-idef"%i, stats, ip.cvx_opt_joint, pdg, also_idef=False)
		# expt.enqueue("%d--cvx+idef"%i, stats, ip.cvx_opt_joint, pdg, also_idef=True)
		ctree_args = dict(varname_clusters=ctree.nodes(), cluster_edges=ctree.edges())

		expt.enqueue("%d--ctree-idef"%i, stats, ip.cvx_opt_clusters, pdg, also_idef=False, **ctree_args, output_processor=pprocessor(pdg))
		expt.enqueue("%d--ctree+idef"%i, stats, ip.cvx_opt_clusters, pdg, also_idef=True, **ctree_args, output_processor=pprocessor(pdg))
		#,verbose=verb
		# collect_inference_data_for(bn_name+"-as-pdg", pdg, store)

		for gamma in args.gammas:
			expt.enqueue("%d--cccp--gamma%.0e"%(i,gamma), stats,
								ip.cccp_opt_clusters, pdg, 
								gamma=gamma, **ctree_args, 
								output_processor=pprocessor(pdg)) #, verbose=verb
			# expt.enqueue(str(i), stats, ip.cccp_opt_joint, pdg, gamma=gamma)
			
			# for ozrname in ['adam', "lbfgs", "asgd"]:
			## don't know what to do here... will use a joint distribution!
			# for ozrname in args.ozrs:
			# 	expt.enqueue("%d--torch(%s)--gamma%.0e"%(i,ozrname,gamma), stats,
			# 	# expt.enqueue(str(i), stats,
			# 		torch_opt.optimize_via_FGs, pdg,
			# 		gamma=gamma, optimizer=ozrname)
				
	
	expt.done()
except (KeyboardInterrupt, InterruptedError) as e:
	print("Interrupted! Dumping results ... ")

except Exception as e:
	print("Uh-oh...", e)
	raise

finally:
	with open(args.datadir+"/RESULTS.json", 'w') as f:
		json.dump([r._asdict() for r in expt.results.values() if r is not None ], f)
	
	print('... finished writing to "RESULTS.json! ')


	# with open("library.pickle", 'w') as f:
	# 	pickle.dump(store, f)
	# print(expt.results)

