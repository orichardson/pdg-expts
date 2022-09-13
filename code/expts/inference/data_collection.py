"""
Given a PDG, collet 
"""

from pgmpy.inference import ExactInference



from pdg.pdg import PDG
from pdg.store import TensorLibrary
from pdg.rv import Variable as Var
from pdg.dist import CPT, RawJointDist as RJD 

def collect_data(id:str,  M:PDG,  store:TensorLibrary=None):
    """
    run a bunch of algorithms to do inference on the given PDG, and
    store the results in the given tensorlibrary (if given).
    Returns a tensor library of results.
    """
    
    if store == None:
        store = TensorLibrary()

        
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