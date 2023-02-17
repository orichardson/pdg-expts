import argparse
parser = argparse.ArgumentParser(description="collect experimental data from Bayesian Networks")
parser.add_argument("--data-dir", dest='datadir', type=str, 
	default='bn-data',
	help="the name of directory to store points in")

example_bn_names =  [ 
	"asia", "cancer", "earthquake", "sachs", "survey", "insurance", "child",
	 "barley", "alarm",
	#  "mildew", "water", "hailfinder", "hepar2", "win95pts", "andes", "diabetes",
	# "link", "munin1", "munin2", "munin3", "munin4", "pathfinder", "pigs", "munin" 
]
parser.add_argument("-z", "--ozrs", nargs='*', type=str,
	# default=['adam', 'lbfgs' ,'asgd'],
	default=['adam', 'lbfgs'],
	help="Which optimizers to use? Choose from {adam, lbfgs, asgd, sgd}")

# parser.add_argument("--idef", action="store_true")

parser.add_argument("--BNs", default=example_bn_names, nargs='*',
	help="do a second optimization to also optimize IDef subject to Inc minimization")
parser.add_argument("-g", "--gammas", nargs='*', type=float,
	# default=[1E-8, 1E-4, 1E-2, 1, 2, 10],
	default=[0, 1E-2, 1],
	help="Selection of gamma values")
parser.add_argument("--num-cores", type=int, default=-1)
parser.add_argument("--verbose", action="store_true", default=False)

args=parser.parse_args()



import json
import numpy as np

from pgmpy.inference import BeliefPropagation
from pgmpy.utils import get_example_model


import sys; sys.path.append("../..")

from pdg.pdg import PDG

# from pdg.rv import Variable as Var
# from pdg.dist import CPT, RawJointDist as RJD, Dist

from pdg.alg import interior_pt as ip
from pdg.alg import torch_opt
import traceback
import sys


from expt_utils import MultiExptInfrastructure


if __name__ == '__main__':
	# main()
# def main():
	expt = MultiExptInfrastructure(args.datadir,  n_threads=args.num_cores)

	for bn_name in example_bn_names:
		bn = get_example_model(bn_name)
		pdg = PDG.from_BN(bn)

		stats = dict(
			graph_id = bn_name,
			n_vars = len(pdg.varlist),
			n_worlds = int(np.prod(pdg.dshape)), # without cast, json cannot interperet int64 -.-
			n_params = int(sum(p.size for p in pdg.edges('P'))), #here also	
			n_edges = len(pdg.Ed)
		)

		try:
			bp = BeliefPropagation(bn)
			expt.enqueue(bn_name+"-belief-prop",stats, bp.calibrate, output_processor=lambda r: (0,0))

		except Exception as ex:
			expt.jobnum += 1
			print("BP failed (probably not connected)", flush=True)
			sys.stderr.write("".join(traceback.TracebackException.from_exception(ex).format()))
				
		expt.enqueue(bn_name+"-as-pdg.ip.-idef",stats, ip.cvx_opt_joint, pdg, also_idef=False)
		expt.enqueue(bn_name+"-as-pdg.ip.+idef",stats, ip.cvx_opt_joint, pdg, also_idef=True)

		for gamma in args.gammas:
			expt.enqueue(bn_name+"-as-pdg.cccp.gamma%.0e"%gamma, stats,
					 ip.cccp_opt_clusters, pdg, gamma=gamma)
			for ozrname in args.ozrs:
				# for oi in niters[ozrname]:
				expt.enqueue(bn_name+".torch(%s).gamma%.0e"%(ozrname,gamma), stats,
					torch_opt.opt_clustree, pdg,
					gamma=gamma, optimizer=ozrname,
					#  iters=oi
					)
				

	expt.done()

	with open(args.datadir+"/RESULTS.json", 'w') as f:
		json.dump(expt.results, f)


