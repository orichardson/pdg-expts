import pandas as pd
import numpy as np

from abc import ABC
from typing import Type, TypeVar, Union, Mapping
import collections

from functools import reduce
from operator import mul

import utils 
import rv

import itertools
# recipe from https://docs.python.org/2.7/library/itertools.html#recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


class CDist(ABC): pass
class Dist(CDist): pass

SubCPT = TypeVar('SubCPT' , bound='CPT')


class CPT(CDist, pd.DataFrame, metaclass=utils.CopiedABC):
    PARAMS = {"nfrom", "nto"}
    _internal_names = pd.DataFrame._internal_names + ["nfrom", "nto"]
    _internal_names_set = set(_internal_names)
    
    
    # def __call__(self, pmf):
    #     pass
    # def __matmul__(self, other) :
    #     """ Overriding matmul.... """
    #     pass
    
    def flattened_idx(self):
        cols = self.columns.to_flat_index().map(lambda s: s[0])
        rows = self.index.to_flat_index().map(lambda s: s[0])

        return pd.DataFrame(self.to_numpy(), columns = cols, index=rows)
        
    
    @classmethod
    def from_matrix(cls: Type[SubCPT], nfrom, nto, matrix, multi=True) -> SubCPT:
        if multi:
            cols = pd.MultiIndex.from_tuples(
                [ (tuple(utils.flatten_tuples(v)) if type(v) is tuple
                    else (v,) ) for v in nto.ordered ],
                names=nto.name.split("×"))
                
            rows = pd.MultiIndex.from_tuples(
                [ (tuple(utils.flatten_tuples(v)) if type(v) is tuple 
                    else (v,) ) for v in nfrom.ordered ],
                names=nfrom.name.split("×"))
                
        else:
            cols,rows = nto.ordered, nfrom.ordered

        return cls(matrix, index=rows, columns=cols, nto=nto,nfrom=nfrom)

    @classmethod
    def from_ddict(cls: Type[SubCPT], nfrom, nto, data) -> SubCPT:
        for a in nfrom:
            row = data[a]
            if not isinstance(row, collections.Mapping):
                try:
                    iter(row)
                except:
                    data[a] = { nto.default_value : row }
                else:
                    data[a] = { b : v for (b,v) in zip(nto,row)}
                    
            total = sum(v for b,v in data[a].items())
            remainder = nto - set(data[a].keys())
            if len(remainder) == 1:
                data[a][next(iter(remainder))] = 1 - total
        
        matrix = pd.DataFrame.from_dict(data , orient='index')
        return cls(matrix, index=nfrom.ordered, columns=nto.ordered, nto=nto,nfrom=nfrom)
        
    @classmethod
    def make_random(cls : Type[SubCPT], vfrom, vto):
        mat = np.random.rand(len(vfrom), len(vto))
        mat /= mat.sum(axis=1, keepdims=True)
        return cls.from_matrix(vfrom,vto,mat)
        
    @classmethod
    def det(cls: Type[SubCPT], vfrom, vto, mapping) -> SubCPT:
        mat = np.zeros((len(vfrom), len(vto)))
        for i, fi in enumerate(vfrom.ordered):
            # for j, tj in enumerate(vto.ordered):
            mat[i, vto.ordered.index(mapping[fi])] = 1
    
        return cls.from_matrix(vfrom,vto,mat)
        # return cls.from_matrix(, index=vfrom.ordered, columns=vto.ordered, nto=vto, nfrom= vfrom)


## useless helper methods to either use dict values or list.
def _definitely_a_list( somedata ):
    if type(somedata) is dict:
        return list(somedata.values())
    return list(somedata)


# define an event to be a value of a random variale.
class RawJointDist(Dist):
    def __init__(self, data, varlist):
        self.data = data
        self.varlist = varlist
        
        if rv.Unit not in varlist:
            self.varlist = [rv.Unit] + self.varlist
            self.data = self.data.reshape(1, *self.data.shape)
        
        self._query_mode = "dataframe" # query mode can either be
            # dataframe or ndarray
        
    # TODO: Make this synatx more useful.
    def __call__(self, evt, given=None):       
        # sum {x_i != evt} p(x,y,z,..., evt)
        val, var = evt
        idx = self._idx(var)
        
        reduced = np.sum(self.data, axis=tuple(i for i in range(len(self.varlist)) if i != idx))
        
        # othervarvals = itertools.product(X for X in varlist if X.name != var)
        # for setting in othervarvals:
        #     total += data[setting[idx:] + [val] setting[:idx]]
        
        return reduced[var.ordered.index(val)]
    
    
    def _proccess_vars(self, vars, given=None):
        if vars is ...:
            vars = self.varlist
            
        if isinstance(vars, rv.Variable) \
            or isinstance(vars, rv.ConditionRequest) or vars is ...:
                vars = [vars]
            
        targetvars = []
        conditionvars = list(given) if given else []

        mode = "join"

        for var in vars:
            if isinstance(var, rv.ConditionRequest):
                if mode == "condition":
                    raise ValueError("Only one bar is allowed to condition")
                    
                mode = "condition"
                targetvars.append(var.target)
                conditionvars.append(var.given)
            else:
                l = (conditionvars if mode == "condition" else targetvars)
                if isinstance(var, rv.Variable):
                    l.append(var)
                    # if mode == "condition":
                    #     conditionvars.append(var)
                    # elif mode == "join":
                    #     targetvars.append(var)
                elif var is ...:
                    l.extend(v for v in self.varlist if v not in l)
                else:
                    raise ValueError("Could not interpret ",var," as a variable")
                
        return targetvars, conditionvars
    
    def _idx(self, var):
        try:
            return self.varlist.index(var)
        except ValueError:
            raise ValueError("The queried varable", var, " is not part of this joint distribution")
    
    def broadcast(self, cpt : CPT, vfrom=None, vto=None) -> np.array:
        """ returns its argument, but shaped
        so that it broadcasts properly (e.g., for taking expectations) in this
        distribution. 
        
        Parameters
        ----
        > cpt: the argument to be broadcast
        > vfrom,vto: the attached variables (supply only if cpt does not have this data)
        """
        if vfrom is None: vfrom = cpt.nfrom
        if vto is None: vto = cpt.nto
        
        idxf = self.varlist.index(vfrom)
        idxt = self.varlist.index(vto)
        
        shape = [1] * len(self.varlist)
        shape[idxf] = len(self.varlist[idxf])
        shape[idxt] = len(self.varlist[idxt])
        
        # print(f,'->', t,'\t',shape)
        # assume cpd is a CPT class..
        # but we don't necessarily want to do this in general
        
        cpt_mat = cpt.to_numpy() if isinstance(cpt, pd.DataFrame) else cpt
        if idxt < idxf:
            cpt_mat = cpt_mat.T

        return cpt_mat.reshape(*shape)

    
    def conditional_marginal(self, vars, query_mode=None):
        if query_mode is None: query_mode = self._query_mode
        # if coordinate_mode is "joint": query_mode = "ndarray"
        
        # print(type(vars), vars, isinstance(vars, rv.Variable))
        targetvars, conditionvars = self._proccess_vars(vars)

        idxt = [ self._idx(var) for var in targetvars ]
        idxc = [ self._idx(var) for var in conditionvars ]
        IDX = idxt + idxc
        
        # sum across anything not in the index
        joint = self.data.sum(axis=tuple(i for i in range(len(self.varlist)) if i not in IDX))
        
        # duplicate dimensions that occur multiple times by 
        # an einsum diagonalization...
        joint_expanded = np.zeros([self.data.shape[i] for i in IDX])
        np.einsum(joint_expanded, IDX, np.unique(IDX).tolist())[...] = joint
        
        if len(idxc) > 0:
            # if idxt is first...
            normalizer = joint_expanded.sum(axis=tuple(i for i in range(len(idxt))), keepdims=True)
            
            #if idxt is last...
            # normalizer = joint_expanded.sum(axis=tuple(-i-1 for i in range(len(idxt))), keepdims=True)
        
            # return joint_expanded / normalizer
            matrix = joint_expanded / normalizer;
            if query_mode == "ndarray":
                return matrix
            elif query_mode == "dataframe":
                vfrom = reduce(mul,conditionvars)
                vto = reduce(mul,targetvars)
                mat2 = matrix.reshape(len(vto),len(vfrom)).T

                return CPT.from_matrix(vfrom,vto, mat2,multi=False)
        else:
            # return joint_expanded
            if query_mode == "ndarray":
                return joint_expanded
            elif query_mode == "dataframe":
                mat1 = joint_expanded.reshape(-1,1).T;
                return CPT.from_matrix(rv.Unit, reduce(mul,targetvars), mat1,multi=False)
                
    # returns the marginal on a variable
    def __getitem__(self, vars):
        return self.conditional_marginal(vars, self._query_mode)
    
    
    def prob_matrix(self, *vars, given=None):
        """ A global, less user-friendly version of 
        conditional_marginal(), which keeps indices for broadcasting. """        
        tarvars, cndvars = self._proccess_vars(vars, given=given)
        idxt = [ self._idx(var) for var in tarvars ]
        idxc = [ self._idx(var) for var in cndvars ]
        IDX = idxt + idxc
        
        N = len(self.varlist)
        dim_nocond = tuple(i for i in range(N) if i not in idxc )
        dim_neither = tuple(i for i in range(N) if i not in IDX ) # sum across anything not in the index
        collapsed = self.data.sum(axis=dim_neither, keepdims=True)
        
        if len(cndvars) > 0:
            # collapsed /= collapsed.sum(axis=dim_nocond, keepdims=True)
            collapsed = np.ma.divide(collapsed, collapsed.sum(axis=dim_nocond, keepdims=True))
            
        return collapsed

            
    def H(self, *vars, base=2, given=None):
        """ Computes the entropy, or conditional
        entropy of the list of variables, given all those
        that occur after a ConditionRequest. """
        
        return - (np.ma.log( self.prob_matrix(*vars, given=given) ) * self.data).sum() / np.log(base)
        
        ## The expanded version looks like this, but is 
        ## a bit slower and not really simpler.
        # collapsed = self.prob_matrix(vars)
        # surprise = - np.ma.log( collapsed ) / np.log(base)
        # E_surprise = surprise.filled(0) * self.data
        # return E_surprise.sum()
    
    def I(self, *vars, given=None):
        tarvars, cndvars = self._proccess_vars(vars, given)
        
        n = len(tarvars)
        sum = 0
        
        for s in powerset(tarvars):
            # print(s, (-1)**(n-len(s)), self.H(*s, given=cndvars))
            sum += (-1)**(len(s)+1) * self.H(*s, given=cndvars) # sum += (-1)**(n-len(s)+1) * self.H(*s, given=cndvars)
        return sum
    
    # def _info_in(self, vars_in, vars_fixed):
        # return self.H(vars_in | vars_fixed)
    # 
    def iprofile(self) :
        """
        Returns a tensor of shape 2*2*2*...*2, one dimension for each
        variable. For example, 
            00000 is going to always have zero.
            01000 is the information H(X1 | X0, X2, ... Xn)
            11000 is the conditional mutual information I(X1; X2 | ...)
            
        """
        for S in powerset(self.varlist):
            pass
    
    
    def info_diagram(self, X, Y, Z=None):
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn3
         
         
        H = self.H
        I = self.I
        
        infos = [I(X|Y,Z), I(Y|X,Z), I(X,Y|Z), I(Z|X,Y), I(X,Z|Y), I(Y,Z|X), I(X,Y,Z) ]
        # infos = [round(i, 3) for i in infos]
        infos = [int(round(i * 100)) for i in infos]
        # Make the diagram
        v = venn3(subsets = infos) 
        return v

    #################### CONSTRUCTION ######################
                
    @staticmethod
    def unif( vars) -> 'RawJointDist':
        varlist = _definitely_a_list(vars)
        data = np.ones( tuple(len(X) for X in varlist) )
        return RawJointDist(data / data.size, varlist)

    @staticmethod
    def random( vars) -> 'RawJointDist':
        varlist = _definitely_a_list(vars)
        data = np.random.rand( *[len(X) for X in varlist] )
        return RawJointDist(data / np.sum(data), varlist)
