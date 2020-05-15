

A PDG is intended to have a semantics which picks a single distribution. Why?
	- For comparison to other graphical models
	- To show a global, glued together semantics of all of the pieces
	- To be usable as a replacement for other models that deal with probabilities

	
What are the different ways one might get at such a unique distribution?
	- Minimizers of the energy landscape semantics
	- The fixed point of a PDG regarded as a DBN. 
	- Local conflict resolution.

	

Note: DNs (Dependency Networks, Heckermann) capture the same set of independences as MRFs. 
	But note that DNs do NOT represent the independences given by a BN. 
	
	In order to get this, we need to correct for the 


MRF -----> Weighted potentials  -------> ExpFam
PDG -----> Certainties on edges -------> WeightedPDG



The underlying graph of a BN and PDG are the same; the PDG just is parameterized
differently.  

A local semantics for PDGs should therefore combine them and produce a cpt. If our PDG semantics does not do this, I suspect it will have global effects that are hard to understand. 
For example:

	  X -----> Z <------- Y

	"Common Effect", "v-structure", or "collider"

	D-separation. A trail X1, ... , Xn is active given Z if
		for every  X_{i-1} ---> X_i <--- X_{i+1}, X_i or a descendent is in Z
		and no other node of the trail is in Z.
		
	Conditioning on a V activates the V structure. 
