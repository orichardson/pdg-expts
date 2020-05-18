# %load_ext autoreload
# %autoreload 2


import pandas as pd
import numpy as np
import networkx as nx

import collections
from numbers import Number

import utils
from rv import Variable, ConditionRequest, Unit
from dist import Dist, CDist, RawJointDist, CPT

# CPT.from_ddict(One, PS, {'*': 0.3})
# V = Variable("abc")X
# CPT.from_ddict(V*V, One, { ('a', 'a') : .2, ('a', 'b') : .1, ('a', 'c') : .3, ('b', 'a') : .4, 
#     ('b', 'b') : .7, ('b', 'c') : .9, ('c', 'a') : .8, ('c', 'b') : .6, ('c', 'c') : .5})

# class LinkView:
#     def __get__(self, instance, cls) -> object:
#         instance.cpd.keys()
# 
#     def __set__(self, obj, value) -> None:
#         pass
# 
#     # def __getitem__()
ln2 = np.log(2)

def joint_index(varlist):
    """Given sets/variables/lists [Unit, A, B, C], reutrns
    np.array([ (⋆, a0, b0, c0), .... (⋆, a_m, b_n, c_l) ]) 
    """
    return np.rollaxis(np.stack(np.meshgrid(*[np.array(*X) for X in varlist],  indexing='ij')), 0, len(varlist)+1)
    
def z_mult(joint, masked):
    """ multiply assuming zeros override nans; keep mask."""
    return np.ma.where(joint == 0, 0, joint * masked)

class Labeler:
    # NAMES = ['p','q','r']
    
    def __init__(self):
        self._counter = 0
        # self._edge_specific_counts = {}
        
    def fresh(self, vfrom, vto, **ctxt):
        self._counter += 1
        return "p" + str(self._counter)
        
class PDG:
    # By default, use the base labeleler, which
    # just gives a fresh label by incrementing a counter.
    def __init__(self, labeler: Labeler = Labeler()):
        self.vars = {} # varname => Variable
        self.edgedata = collections.defaultdict(dict)
            # { (nodefrom str, node to str, label) => { attribute dict } }
            # attributes: cpt, alpha, beta. context. (possible lambdas.). All optional.

        self.labeler = labeler
        self.graph = nx.MultiDiGraph()
        self.gamma = 1
        
    # generates <node_name_from, node_name_to, edge_label> 
    # @property
    # def E(self, include_cpts = False) -> Iterator[str, str, str, Number]:
    #     for i,j in self.cpds.keys():
    #         if include_cpts:
    #             for l, L in cpts[i,j].items():
    #                 yield (i,j,l), L
    #         else:
    #             for l in cpds[i,j].keys():
    #                 yield i,j,l
        
    def copy(self):
        rslt = PDG(self.labeler)
        
        # rslt.vars = dict(**self.vars) # variables don't need a deep copy. 
        
        for vn, v in self.vars.items():
            rslt._include_var(v,vn)
        
        for ftl, attr in self.edgedata.items():
            rslt._set_edge(*ftl, **attr)

        rslt._apply_structure();
        return rslt
    
    
    @property
    def varlist(self):
        return list(X for X in self.vars.values())
        
    @property
    def dshape(self):
        return tuple(len(X) for X in self.varlist)
    
    def _get_edgekey(self, spec):
        label = None
        if isinstance(spec, ConditionRequest):
            gn,tn = spec.given.name, spec.target.name
            # raise KeyError("Multiple options possible.")
        elif type(spec) == tuple and type(spec[0]) is str:
            # normal strings can be looked up as a tuple
            gn,tn = spec[:2]
            if len(spec) == 3:
                label = spec[2]          
        elif type(spec) is str:
            for xyl in self.edgedata:
                if ','.join(xyl) in spec or spec is xyl[-1]:
                    gn, tn, label = xyl
                    
        return gn,tn,label        
        
    def with_params(self, **kwargs):
        rslt = self.copy()
        for param, val in kwargs.items():
            if type(val) is dict:
                for spec, defn in val.items():
                    self.edgedata[self._get_edgekey(spec)][param] = defn
        return rslt
    
    def _apply_structure(self, focus=None):
        if focus is None:
            focus = self.vars
        
        for f in focus:
            var = self.vars[f] if type(f) is str else f
                
            for vstructure in var.structure:
                for relation in vstructure.gen_cpts_for(self):
                    self += relation
    
    # should only need to use this during a copy.
    # def _rebuild_graph(self):
    #     for vn, v in self.vars.items():
    #         self.graph.add_node(vn, {'var' : v})
    # 
    #     for (f,t,l), attr in self.edgedata:
    #         self.graph.add_edge(f,t, key=l, attr=dict(label=l,**attr))
        
    def _include_var(self, var, varname=None):
        if varname:
            if not hasattr(var,'name'):
                var.name = varname
            assert varname == var.name, "Variable has thinks its name is different from the PDG does..."
        
        if not hasattr(var,'name') or var.name == None:
            raise ValueError("Must name variable before incorporating into the PDG.")
        

        self.graph.add_node(var.name, var=var)
        if var.name in self.vars:
            self.vars[var.name] |= var.ordered
        else:
            self.vars[var.name] = var
            self._apply_structure([var])
                

    def _set_edge(self, nf: str, nt: str, l, **attr):
        edkey = self.graph.add_edge(nf,nt,l, **attr)
        self.edgedata[nf,nt,edkey] = self.graph.edges[nf,nt,edkey]
        

    def add_data(self, data, label=None):
        """ Include (collection of) (labeled) cpt(s) or variable(s) in this PDG.
        
        Adding a PDG itself computes a union; one can also add an individual
        cpt, or a variable, or a list or tuple of cpts.    
        
        Can be given labels by
        >>> M: PDG  += "a", p : CPT     
        """           
        
        if isinstance(data, PDG):
            for varname, var in data.vars.items():
                self._include_var(var,varname)
                
            for ftl, attr in data.edgedata.items():
                self._set_edge(*ftl, **attr)
            
        elif isinstance(data, CPT):
            self._include_var(data.nfrom)
            self._include_var(data.nto)
            # label = other.name if hasattr(other, 'name') else \
            #     self.labeler.fresh(other.nfrom,other.nto)
            
            self._set_edge(data.nfrom.name, data.nto.name, label, cpd=data)
        
        elif isinstance(data, Variable):
            vari = data.copy()
            if label and not hasattr(vari, 'name'):
                vari.name = label
            self._include_var(vari)
        
        elif type(data) in (tuple,list):
            for o in data:
                self.add_data(o, label)
        else :
            print("Warning: could not add data", data, "to PDG")
            return -1
                                
        
    def __iadd__(self, other):
        if type(other) is tuple and len(other) > 1 and type(other[0]) is str:            
            self.add_data(other[1], label=other[0])
            self += other[2:]
        else:
            self.add_data(other)
            
        return self

        
    def __add__(self, other):
        rslt = self.copy()
        rslt += other;
        return rslt
        # return (self.copy() += other)
    
    def __delitem__(self, key):
        if isinstance(key, tuple):
            k = list(key)
            if len(key) in [2,3]:
                if isinstance(key[0], Variable):
                    k[0] = key[0].name
                if isinstance(key[1], Variable):
                    k[1] = key[1].name
                    
                if(len(key) == 2):
                    k += [next(iter(self.graph[k[0]][k[1]].keys()))]
                    
                self.graph.remove_edge(*k)
                del self.edgedata[tuple(k)]
                
        if isinstance(key, str):
            if key in self.vars:
                del self.vars[key]
                self.graph.remove_node(key)

    # For convenience only. 
    # takes a pair (src, target) of variables, and returns the relevant cpt. 
    # Alternatively, takes a string name   
    def __getitem__(self, key):
        label = None
        if isinstance(key, ConditionRequest):
            gn,tn = key.given.name, key.target.name
            # raise KeyError("Multiple options possible.")
        elif type(key) == tuple and type(key[0]) is str:
            # normal strings can be looked up as a tuple
            gn,tn = key[:2]
            if len(key) == 3:
                label = key[2]            
        else:
            try:
                gn,tn,label = self._get_edgekey(key)
            except e:
                print(e) 
                print(key, 'is not a valid key')
                return

        if label == None:
            if len(self.graph[gn][tn]) == 1:
                return next(iter(self.graph[gn][tn].values()))['cpd']
            
            return self.graph[gn][tn]['cpd']
        
        return self.edgedata[gn,tn,label]['cpd']
        # if (obj == )
        
    def __iter__(self):
        for (Xname, Yname, l), data in self.edgedata.items():
            X,Y = self.vars[Xname], self.vars[Yname]
            alpha = data.get('alpha', 1)
            beta = data.get('beta', 1)
            
            yield X,Y, data.get('cpd', None), alpha, beta
        
    # semantics 1:
    def matches(self, mu):
        for X,Y, cpd, *_ in self:
            # print(mu[Y], '\n', mu[X], '\n', cpd)
            if( not np.allclose(mu[Y], mu[X] @ cpd) ):
                return False

        return True
        
    
    def _build_fast_scorer(self, weightMods=None, gamma=None):
        N_WEIGHTS = 4
        if weightMods is None:
            weightMods = [lambda w : w] * N_WEIGHTS
        else:
            weightMods = [
                    (lambda n: lambda b: n)(W) if isinstance(W, Number) else
                    W if callable(W) else 
                    (lambda b: b)
                for W in weightMods]
            
        if gamma == None:
            gamma = self.gamma
            
        weights = np.zeros((N_WEIGHTS, len(self.edgedata)))
        for i,(X,Y,cpd, alpha,beta) in enumerate(self):
            w_suggest = [beta, -beta + alpha*gamma, 0, 0]
            for j, (wsug,wm) in enumerate(zip(w_suggest, weightMods)):
                weights[j,i] = wm(wsug)

        SHAPE = self.dshape
        mu = RawJointDist.unif(self.varlist)
        Pr = mu.prob_matrix
                    
        def score_vector_fast(distvec):
            # enforce constraints here... 
            distvec = np.abs(distvec).reshape(*SHAPE)
            distvec /= distvec.sum() 
            # mu = RawJointDist(distvec, self.varlist) 
            mu.data = distvec # only create one object...
                        
            PENALTY = 1001.70300201
                # A large number unlikely to arise naturally with
                # nice numbers. 
                # Very hacky but faster to do in this order.

            thescore = 0        
            for i, (X,Y,cpd_df,alpha,beta) in enumerate(self):
                # This could easily be done 3x more efficiently
                # by generating the matrices jointly.
                # look here if optimization reqired. 
                muy_x, muxy, mux = Pr(Y | X), Pr(X, Y), Pr(X)
                cpt = mu.broadcast(cpd_df)            
                logcpt = - np.ma.log(cpt) 
                                            
                logliklihood = z_mult(muxy, logcpt.filled(PENALTY))
                logcond_info = z_mult(muxy, -np.ma.log(muy_x).filled(PENALTY) )
                logextra = z_mult(mux * cpt, logcpt.filled(PENALTY))
                                            
                ### The "right" thing to do is below.
                # logliklihood = z_mult(muxy, logcpt)
                # logcond_info = z_mult(muxy, -np.ma.log(muy_x) )
                # logextra = z_mult(mux * cpt, logcpt)
                
                ### Dependency Network Thing.
                # nonXorYs = [Z for Z in self.vars.values() if Z is not Y and Z is not X]
                # dnterm = z_mult(muxy, -np.ma.log(Pr(Y|X,*nonXorYs)))
                
                # no complex, just big score, in fast version
                thescore += weights[0,i] * logliklihood.filled(PENALTY).sum()  
                thescore += weights[1,i] * logcond_info.filled(PENALTY).sum()  
                thescore += weights[2,i] * logextra.filled(PENALTY).sum()  
                    

            ################# λ4 ###################
            thescore /= np.log(2)
            thescore -= gamma * mu.H(...)
            
            return thescore
        return score_vector_fast
    
    #  semantics 2  
    def score(self, mu : RawJointDist, weightMods=None):
        if weightMods is None:
            weightMods = [lambda w : w] * 4
        else:
            weightMods = [
                (lambda n: lambda b: n)(W) if isinstance(W, Number) else
                W if callable(W) else 
                (lambda b: b)
                    for W in weightMods]
        
        thescore = 0
        infoscores = np.zeros(mu.data.shape)
        
        for X,Y,cpd_df,alpha,beta in self:
            # This could easily be done 3x more efficiently
            # look here if optimization reqired. 
            Pr = mu.prob_matrix
            muy_x, muxy, mux = Pr(Y | X), Pr(X, Y), Pr(X)
            cpt = mu.broadcast(cpd_df)            
            logcpt = - np.ma.log(cpt) 
                        
            weights = [beta, -beta, alpha*self.gamma, 0]
            
            ### Liklihood.              E_mu log[ 1 / cpt(y | x) ]
            logliklihood = z_mult(muxy, logcpt)
            ### Local Normalization.    E_mu  log[ 1 / mu(y | x) ]
            logcond_info = z_mult(muxy, -np.ma.log(muy_x) )
            ### Extra info.         E_x~mu cpt(y|x) log[1/cpt(y|x)]
            logextra = z_mult(mux * cpt, logcpt)
            ### Dependency Network Thing.
            nonXorYs = [Z for Z in self.varlist if Z is not Y and Z is not X]
            dnterm = z_mult(muxy, -np.ma.log(Pr(Y|X,*nonXorYs)))
            
            
            terms = [logliklihood, logcond_info, logextra, dnterm]
            for term, λ, mod in zip(terms, weights, weightMods):
                # adding the +1j here lets us count # of infinite
                # terms. Any real score is better than a complex ones
                thescore += mod(λ) * (term).astype(complex).filled(1j).sum()
                infoscores += mod(λ) * term # not an issue here
                

        ################# λ4 ###################
        thescore /= np.log(2)
        thescore -= self.gamma * mu.H(...)
        
        return thescore.real if thescore.imag == 0 else thescore
        

    ####### SEMANTICS 3 ##########
    # WARNING: this will output the distribution 
    def optimize_score(self):
        scorer = self._build_fast_scorer()
        factordist = self.factor_product().data.reshape(-1)
        init = RawJointDist.unif(self.vars).data.reshape(-1)
        # alternate start:
        # init = factordist
        
        from scipy.optimize import minimize, LinearConstraint, Bounds

        req0 = (factordist == 0) + 0
        opt1 = minimize(f, 
            init,
            constraints = [LinearConstraint(np.ones(init.shape), 1,1), LinearConstraint(req0, 0,0.01)],
            bounds = Bounds(0,1),
                # callback=(lambda xk,w: print('..', round(f(xk),2), end=' \n')),
            method='trust-constr',
            options={'disp':False}) ;
        rslt = minimize(scorer, init)
        self._opt_rslt = rslt
        
        varlist = self.varlist
        rsltdata = abs(rslt.x).reshape(self.dshape)
        rsltdata /= rsltdata.sum()
        
        return RawJointDist(rsltdata, varlist)
        # TODO: figure out how to compute this. Gradient descent, probably. Because convex.
        # TODO: another possibly nice thing: a way of testing if this is truly the minimum distribution. Will require some careful thought and new theory.


    ##### OTHERS ##########
    def factor_product(self) -> RawJointDist:
        # start with uniform
        d = RawJointDist.unif(self.vars)
        for X,Y,cpt,*_ in self:
            if cpt is not None:
                #hopefully the broadcast works...
                d.data *=  d.broadcast(cpt)
        d.data /= d.data.sum()
        
        return d

    ############# Testing 
    def random_consistent_dists():
        """ Algorithm:  
        """
        pass
