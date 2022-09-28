import sys
sys.path.append('../..')

from pdg import *
from pdg.alg import interior_pt as ip
from pgmpy.utils import get_example_model


from time import time

import argparse
parser = argparse.ArgumentParser(description="Run pdg.alg.interior_pt.cvx_opt_cluster algorithm on the given Bayesian Network")
parser.add_argument("--bn", metavar='bn', type=str, help="the name of the BN from the bnlearn repository.")
parser.add_argument("--also-idef", help="do a second optimization to also optimize IDef subject to Inc minimization",
                    action="store_true")

args=parser.parse_args()


print("loading model %s"%args.bn)

bn = get_example_model(args.bn.lower())
pdg = PDG.from_BN(bn)

init_time = time()
print( "beginning at ", init_time)
rslt = ip.cvx_opt_clusters(pdg, also_idef=args.also_idef, verbose=True)

cd = rslt.cluster_dist

print('='*60)
print("DONE!; took ", time() -init_time)
print(rslt.inc)
print(rslt.idef)
print(rslt)
