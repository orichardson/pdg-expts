# Section 1 [Introduction]
Same as overleaf, except  I still want to drop a hint or two about models that are not BNs when we go through examples.

# Section 2 [Formalism and Syntax]
Same as on overleaf except without \alpha, and with example 5 deleted.
Perhaps also make \beta part of the main definition; I don't see a good reason to introduce the term "weighted"; so many other modifiers (strict, qualitative, etc.,) are going to be more important later, and I don't see a good use case for explicitly not wanting weights instead of default 1.

# Section 3 [Semantics]
Introduce the three semantics, more compactly, without \alpha.
In particular, much less discussion about KL divergence and the extra info.
Change semantics 3 so that it has $\lim_{\gamma -> 0}$

Mention BOTH formulations of extra info to set up later discussion and comparisons. In more detail:

State that [forulation 1] corresponds to the correct qualitative picture without going into details; settle on [formulation 2] for the definition here, because the simpler presentation [formulation 1] properly requires more attention to the qualitative half, which is beyond the scope of the paper. 
Can be further justified by appeal to convexity, interpretation as "extra info beyond constraints", and by equivalence as \gamma->0 for consistent PDGs.


# Section 4 [Relations to other Graphical Models]

## Sec 4.1 [Bayseian Networks]
Mention that the result works for _either_ formulation mentioned in section 3, and ALSO for any value of \gamma or \beta. 

## Sec 4.2 [Factor graphs and MRFs]
Compress and totally reorganize; new story is:

Brief introduction to MRFs (represented by factor graphs). 
Show how letting \gamma=1 gives the factor graphs (in formulation1).
Quickly list a bunch of negative features of factor graphs, and explain why PDGs when \gamma-> 0 do not have any of this behavior.

Show the graph which reveals BNs to be at the  and note that this framework allows us to capture both.

## Sec 4.3 [Other models and related work]
Here we can just quickly overview a lot of related work.
High priority are directed factor graphs, dependency networks, which we should mention but not dwell on because they're esoteric. 


# Section 5 [A Mental Representation]
Reiterate that PDGs have uses beyond specifying distributions; go through the belief updating example to show how inconsistency and modularity work together to model mental state. Emphasize that the modifications to the arrows enable this kind of thing.

Simulate resolutions to inconsistency if we have time; I think we can do this in very little space.

# Section 6 [Conclusions]
Summarize, emphasizing the generality, ability to capture inconsistency, and ease of use over other models. 
Gesture at qualitative PDGs and causality, and non-strict PDGs.