import argparse
parser = argparse.ArgumentParser(description="collect experimental data from Bayesian Networks")
parser.add_argument("--data-dir", dest='datadir', type=str, 
	default='bn-data',
	help="the name of directory to store points in")

example_bn_names =  [ 
	"asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
	# "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts", "andes", "diabetes",
	# "link", "munin1", "munin2", "munin3", "munin4", "pathfinder", "pigs", "munin" 
]

parser.add_argument("--idef", action="store_true")

parser.add_argument("--BNs", default=example_bn_names, nargs='*',
	help="do a second optimization to also optimize IDef subject to Inc minimization")
args=parser.parse_args()



import json
import numpy as np

from pgmpy.inference import BeliefPropagation
from pgmpy.utils import get_example_model

from pdg.pdg import PDG
from pdg.store import TensorLibrary

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
	store = TensorLibrary()

	expt = MultiExptInfrastructure(args.datadir)

	for bn_name in example_bn_names:
		bn = get_example_model(bn_name)
		pdg = PDG.from_BN(bn)

		stats = dict(
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
		# collect_inference_data_for(bn_name+"-as-pdg", pdg, store)

		for gamma in [1E-12, 1E-8, 1E-4, 1E-2, 1, 2]:
			expt.enqueue(bn_name+"-as-pdg.cccp.gamma%.0e"%gamma, stats,
					 ip.cccp_opt_joint, pdg, gamma=gamma)
			
			for ozrname in ['adam', "lbfgs", "asgd"]:
				expt.enqueue(bn_name+"-as-pdg.torch.gamma%.0e"%gamma, stats,
					torch_opt.opt_dist, pdg,
					gamma=gamma, optimizer=ozrname)
				

	expt.done()

	# with open("library.pickle", 'w') as f:
	# 	pickle.dump(store, f)
	with open("RESULTS.json", 'w') as f:
		json.dump(expt.results, f)


