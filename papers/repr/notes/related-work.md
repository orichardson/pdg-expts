# Other Representations
## Bayesian Networks
Perfectly emulated by PDGs ($\beta \mapsto \Gamma \beta$), identical except for (1) merges,  (2) cycles, and (3) independence assumptions.

Merges and cycles don't occur in BNs, and the BN independence assumptions are equivalent to maxent semantics.

**Theorem:** For any BN $\beta$, $[[\Gamma \beta ]]_{H^\uparrow} = \Pr_\beta$

**Notes:**

 - way less flexible with adding / removing info. Tables are bad, densities are worse.
 - Can (possibly?) emulate part of it with the multiple-copy construction--- but then what does it even mean to emulate a variable? What do you mean by an inconsistency?
 - It is possible to marginalize out variables (so functions can always take fewer variables) by variable elimination / sum-product, but not to take unions of sets of variables. Requires more indep. assumptions and impossible to represent.



## Conditional BNs
Also perfectly represented by PDGs, and visually distinct without colors. Moreover, adding and deleting knowledge dynamically updates the type of the PDG but not of the conditional BN. Still inflexible in arrows;

### Dynamic BNS


**Theorem**: Given a 2-DBN ($\beta, \tau$) where $\beta$ is a

## Factor Graphs
## MRFs
Less expressive than factor graphs: can't distinguish between effects of different clique sizes (to emulate cliques).

## CRFs

## Directed Factor Graphs
## Dependency Networks
## Chain Networks
## Wiring Diagrams (Jacobs)


## UNIQUE
Paths mean something.


# Sub-stochastisity

Why are PDG's uniquely posed to take advantage of it?

 * multiple arrows with same target give refinements on a convex hull.
 * Makes it possible to implement plate models.


# Other notes

* You have to
* Sampling BNs / variable eliminations / sum-product not really thought of as matrix multiplications :(

	- Eigenanalysis of
	- Information chanel analysis?

 * Note: information flowing through a BN is equivalent to the trace of an execution of a stochastic program.
