# Defn Wishlisth

- Differentiable information theoretic defn of inconsistency that lines up with the assign-joint-probability one

# Theorem Wishlist

 - [!false] Utility representation: if you can represnt all preferences by utility functions, then there is a way to combine them such that decision making is the same. Ways to formalize "decision making the same"
	- Probability of choice in each set is the same as before.

 - admits utility function <==> consistency

 - Preference consistency is convex: taking mixtures of compositions is always consistent. Moreover, with high probability /accuracy we can represent the truth as a linear combination of compositions, in the limit as we incorporate more variables.

# General Preditions



***** All diagrams having limits => probability distributions. All diagrams having colimits => utility function. Weaker version: final 

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
		
	
# High Level
 - always consistent is not convex. This is a convex realaxation of classical decision making, which has different emergent properties for cognitively bounded agents
 
 
# Future
- inconsistency as a measure on the joint distributions that are consistent


# Historical Log
[~unimportant]
Ignoring beliefs that are not utilities (i.e., no relationships between anything that's not the special utility domain), we can always make a global joint utility that marginalizes for each variable like its original utility. 
 
... but now beliefs with 0 and 1 components can make the total picture inconsistent. This is a good reason for a smoother version of inconsistency, based on information: maybe it's not inconsistent yet but just extremely unlikely? Maybe it was just luck that another beleif wasn't epsilon off and so the difference between consistency and inconsistency is not as clear?
   
