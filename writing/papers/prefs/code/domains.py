import numpy as np
import pandas as pd

from collections import defaultdict
from functools import partial
#from scipy.linalg import expm

class Dom(object):
    def __init__( self, comparison_df):
        self.comp = comparison_df
        
    def __len__(self):
        return len(self.comp.index)
        
    def __iter__(self):
        return iter(self.comp.index)
        
    def __repr__(self):
        return repr(self.comp)
    
    ######################## Constructions ################    
    @classmethod
    def empty( cls, objects ):
        """
        Create an empty domain, where no two objects are comparable (nans)
        """
        return cls( pd.DataFrame(index=objects, columns=objects))
        
    @classmethod
    def indif( cls, objects ):
        """ 
        Create an empty domain with indiference about preferences
        """
        return cls( pd.DataFrame(data=np.zeros((len(objects),)*2), index=objects, columns=objects))
        
    @classmethod
    def from_util(cls, util_dict):
        rslt = defaultdict(dict)
        for x, Ux in util_dict.items():
            for y, Uy in util_dict.items():
                rslt[x][y] = Uy - Ux
                
        return cls(pd.DataFrame(rslt))
    
    @classmethod
    def from_order( cls, objects ):
        rslt = defaultdict(dict)
        for x in objects:
            for y in objects:
                rslt[x][y] = (x < y)
                
        return cls(pd.DataFrame(rslt))
        
    @classmethod
    def from_list_order( cls, objects, certainty=0):
        rslt = defaultdict(lambda: defaultdict(lambda: 0))
        for i,r in enumerate(objects):
            for j,c in enumerate(objects[i:], i):
                rslt[r][c] = 1 - certainty
                rslt[c][r] += certainty -1
                
        return cls(pd.DataFrame(rslt))
        
    @classmethod
    def randu( cls, name, number):
        utildict = { name[0].lower()+"_"+str(i) : np.random.normal()
                              for i in range(number) }
        return cls.from_util(utildict)
        
    ########################  Ensure Properties  ###########################


class Link(object):
    def __init__( self, domain, codomain, Pr): #takes a dataframe
        self.Pr = Pr
        self.dom = domain
        self.cod = codomain
        
    #################### Constructions ##############
    @classmethod
    def from_dict( cls, d, dom = None ):
        dd = defaultdict(dict)
        for k, v in d.items():
            dd[k][v] = 1
            
        pdargs = dict(index=list(set(dom) | set(d.values()))) if dom else {}
        Pr = pd.DataFrame(dd, **pdargs).fillna(0)
        
        return cls(Pr.index, Pr.columns, Pr )
    
    @classmethod
    def from_ddict( cls, dictdict ):
        Pr = pd.DataFrame(dictdict).fillna(0)
        return cls(Pr.index, Pr.columns, Pr )
        
    @classmethod
    def from_dfun( cls, f, A : Dom ):
        return cls.from_dict({ a : f(a) for a in A })
        
    @classmethod
    def by(cls, target_domain=None, **kwargs):
        return cls.from_dict(kwargs, dom=target_domain)

    # algebra
    def rebind(A, B): # rebind to new domains
        pass

## Transformers:
def r_transitive( df ):
    return lambda X: X + X @ df

def l_transitive( df ):
    return lambda X: X + df @ X

def symmetric( df ):
    return lambda X: X


def star( df , revalidate = lambda X : X, stopAfter=1000):
    rslt = df
    for i in range(stopAfter):
        rnew = revalidate(rslt + rslt @ df)
        if np.allclose( rnew, rslt):
            return rnew
        rslt = rnew
    
    print("Before timeout: ", rslt)
    raise ValueError("Closure Computation Timed Out")
        
    

def stoch(X):
    return X / np.sum(X, axis=1)[:, None]
    
############# TESTS ########
# D = make_comp( "ABC", 0)
# 
# A = Dom.from_list_order(list("ABCDE")).comp
# A
# closure(A, stoch)
# 
# D + D @ D
# 
# 
# nD = make_comp("CBA", 0)
# uncertain = make_comp("ABC", 0.5)
