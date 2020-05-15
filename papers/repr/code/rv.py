import utils
import abc

class RV(abc.ABC):
    pass
    # @abc.abstractproperty
    # def vals(self):

class ConditionRequest(object, metaclass=utils.CopiedType):
    PARAMS = {"target", "given"}   

    
    
class Variable(set, metaclass=utils.CopiedType):
    PARAMS = {'name', 'default_value'}
    
    def __init__(self, vals):
        # super init inserted by metaclass
        self._ordered_set = list(vals)
        self.structure = []
        
    def __mul__(self, other):
        kwargs = {}
        if hasattr(self, 'default_value') and hasattr(other, 'default_value'):
            kwargs['default_value'] = (self.default_value, other.default_value)
        if hasattr(self, 'name') and hasattr(other, 'name'):
            kwargs['name'] = (self.name + "×" + other.name)
            
        joint =  Variable([(a,b) for a in self.ordered for b in other.ordered ], **kwargs)
        joint.structure = [*self.structure, *other.structure, JointStructure(joint, self, other)]
        
        return joint
        
    def __ior__(self, other):
        newelts = [ o for o in other if not o in self ]
        self.update(newelts)
        self._ordered_set = self.ordered + newelts
        return self
        
    # with a variable V taking v, can write
    # V.v

    
    """ conditioning """
    def __or__(self, other):
        return ConditionRequest(target=self,given=other)
    
    @property
    def ordered(self):
        self._ordered_set = [x for x in self._ordered_set if x in self] + \
            [y for y in self if y not in self._ordered_set]
        return self._ordered_set
        
    # @property
    # def pd_index(self):
    #     pass
        
    @classmethod
    def alph(cls, name : str, n : int):
        nl = name.lower()
        return cls([nl+str(i) for i in range(n)], default_value=nl+"0", name=name)    
# V = Variable([3, 10, 2], name='V')
# (V*V).name




def binvar(name : str) -> Variable:
    nl = name.lower()
    return Variable([nl, "~"+nl], default_value=nl, name=name)

Unit = Variable('⋆', default_value='⋆', name='1')


class JointStructure: 
    def __init__(self, both, left, right):
        self.joint = both
        self.left = left
        self.right = right
    
    def gen_cpts_for(self, pdg):
        from dist import CPT
        
        if self.joint.name in pdg.vars:
            hasL = self.left.name in pdg.vars
            hasR = self.right.name in pdg.vars
             
            if hasL:
                yield "π1", CPT.det(self.joint, self.left, {v: v[0] for v in self.joint})
            if hasR:
                yield "π2", CPT.det(self.joint, self.right, {v: v[1] for v in self.joint})
                
            # Maybe also: universal property
            # generate CPT going into joint for every pair
            # going into CPT, from any other variable.
            # TODO later
