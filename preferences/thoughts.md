# Problems with CP networks:

- CP networks rely heavily on the fact that all variables "exist" --- that is to say, that each variable takes on particular values, and that they can be assigned

	:	[Question: can you prefer an inconsistent world?]

- As a correlary, counterfactuals are assumed to be extremely simple (flipping a variable is a reaasonable thing to do)
- The formulations presented are only explored in the case of binary variables, and mostly only with acyclic graphs.
- "Twisting" knowledge representation, i.e., creating a new variable to "untangle" a dependency, seems to be an alternate interpretation of what exactly can be done.




Essentially, we have put a topology / structure on the space of possible worlds, and are exploring paths through it. But assuming you have such a structure where all valid assignments of variables are possible is already problematic for the physical world.

Restricting to cases where the domains are well-known: consider user interfaces, and button position.


# Interactions.

It is very important to figure out how we want preferences + meta-preferences to interact. One mechanism is by resources: you only have so much computation, so you have to trade off preference changing against planning, exploring, etc.


How are we going to model this? 



# § 


# § Applications.

## Understanding Humans better
This is the goal of microeconomics. Implications in terms of getting peopel to help follow through their goals [self help],
and related implications for how organization systems ought to be built. 

Examples:

 - Serving content that is better for people
	 - Spotify: give people music that they would like to know better, in addition to what they like.
	 -  Netflix: give people things that they would want to be more cultured, rather than just what the click on.


 - Organizational systems: give reminders about the right things at the right times: birthdays, things you ened to do. The general motto is: help me achieve my meta-preferences, not my actual preferences. 

	For instance, 
 


Explanation of the legal system in terms of preferences + meta preferences. Possible societal / aggregation interpreation. Voting math is maybe a stretch?

## Reinforcement Learning

Design systems that have meta-preferences in addition to preferences, which form from preferneces + experiences, and have ways of correcting preferences.


Connection to adversarial models: avoid overfitting to objective function because objective function (meta-preferences) have an adversarial relationship.



# Formalization Sketches

$\Omega : \mathbf{Set}$

$\mathcal X_0 : \mathbf{Set}$  
$D_0 : \Omega^{\cal X_0}$  
$\leq_0 : \mathcal O(D_0)$

All together, we have $P_0 = (D_0, \leq_0)$, or
\[P_0 : \sum_{D_0 : \Omega^{\mathcal X}} \mathcal O(D_0) \]

----

But for meta-preferneces, we have 
 
