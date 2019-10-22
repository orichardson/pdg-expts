# To DO

- Abstraction: show that we can do matrices with "sub-nodes". Matrix factorization, different semirings, etc. all work.

- TODO: automate all of this. This is definitely something that can (and should!!) have attached code. It will help me find bugs, find the real emergent behavior, etc. etc. It'll be easier to explore and have empirics if you do it right. Also, it'll be fun. And you've already done part of it.


# To Investigate
 - Equalizers
 - Strong / weak silence and conservatieness: what do these mean in terms of theory
 - Savage's axioms: Interpret.
 - Variational Bounds. Principle of free energy. Relate to Karl Friston.
 - Embed Logics: see how perfectly deterministic things of interest play out and what I need to do to accomodate them.

# Theorem Wishlist
 - p ∈ Center of consistent diagram ⟹ preferences are adaptive. 
     That is, utilities are normalized. If X occurs most of the time, it is close to zero utility.

 - admits utility function <==> consistency

 - Preference consistency is convex: taking mixtures of compositions is always consistent. Moreover, with high probability /accuracy we can represent the truth as a linear combination of compositions, in the limit as we incorporate more variables.

- [???ehhh] All diagrams having limits => probability distributions. All diagrams having colimits => utility function.  

# Examples / Observations.

**1**. Suppose you think A is good and B is bad. You think A makes B less likely. Then you realize that A actually makes B more likely. What do you do?
	* Think A is worse than you thought
	* Think B is better than you thought
	
	Classical picture: underlying utilities are the same; expected utility of A is now lower.

	
**3**. Thought experiment: by utility in real life we mean expected utility.
 	Suppose you really like $A$ (ice cream), and assign high utility to it. But now your family and government collaborate to make sure that whenever you eat ice cream you get an unpleasant electric shock, and feel awful for a day. In this new world, do you still like ice cream?  What if instead of an external device, it was merely re-wired to your brain. Would you say you still like ice cream now?
		No! Ice cream is many things. The taste is good, the nutrition is bad, etc. 
	What about just the taste? Do you still like that?
		No! It's associated with pain.
		
	When you can't separate two effects, there's no reason to talk about them, and no way to differentiate between them
		
	
# Fuzzy Intuition
 - always consistent is not convex. This is a convex realaxation of classical decision making, which has different emergent properties for cognitively bounded agents
 
  
# Less Important Things
 - Can we generalize to other Renyi Entropies? In particular, the conservativeness afforded by min-entropy might have a reasonable interpretation, and when substituted into the divergence, may make consistency resolution different.
 
 - inconsistency as a measure on the joint distributions that are consistent

------------------------------------------------------



# Historical Log
[~unimportant]
Ignoring beliefs that are not utilities (i.e., no relationships between anything that's not the special utility domain), we can always make a global joint utility that marginalizes for each variable like its original utility. 
 
... but now beliefs with 0 and 1 components can make the total picture inconsistent. This is a good reason for a smoother version of inconsistency, based on information: maybe it's not inconsistent yet but just extremely unlikely? Maybe it was just luck that another beleif wasn't epsilon off and so the difference between consistency and inconsistency is not as clear?
   

- [!false] Utility representation: if you can represnt all preferences by utility functions, then there is a way to combine them such that decision making is the same. Ways to formalize "decision making the same"
- Probability of choice in each set is the same as before.
