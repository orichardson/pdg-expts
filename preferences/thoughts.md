


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


We also want an agent embedded in an environment, and we want to model the preferences and changes of state as 


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

Do inverse reinforcement learning at the same time as normal reinforcement learning. Make 

# Formalization Sketches

$\Omega : \mathbf{Set}$

$\mathcal X_0 : \mathbf{Set}$  
$D_0 : \Omega^{\cal X_0}$  
$\leq_0 : \mathcal O(D_0)$

All together, we have $P_0 = (D_0, \leq_0)$, or
\[P_0 : \sum_{D_0 : \Omega^{\mathcal X}} \mathcal O(D_0) \]

----

But for meta-preferneces, we have 
 


-----

- Cartesian, non-embedded agents have the same preferences as meta-preferences
- Can the development of preferences be attributed to Hebbian dynamics? Correlation causing things to happen?



Given a constraint, imagine the "typical worlds" satisfying this constraint. Sometimes have some implicit odering between typical worlds. This is more likely if the constraints are structurally similar --- involve the same variables, etc.

	"Imagine a world where pizza is illegal". is this preferable to a world where "shoes are worn on ears". Maybe have an initial preference, followed by some simulation, world-building, and imagination in both worlds. Then preferences over these things change.
	
	But why do they change? Do we already have some implicit preference over simulated worlds, and are trying to get to that? How did we get such a preference? How does it change when the representation of the world and how it evolves gets better?
	
	It is formed by example / experiences, obviously. This is in turn bootstrapped by a biological reward: (pain / pleasure), various experiences with phenomena, which are then fit to a view of the external world. So we learn:
		Hot --> Bad,
		Sugar --> Good
		
		or better:
		Touching hot things --> bad.
		Eating sugar --> good
		
		But they're context dependent. Sometimes it's ok to touch hot things (wearing gloves, have to weigh against things like moving a soldering iron off of your child) or bad to eat sugar. Figuring out the appropriate context is a learning problem; the context is part of the information we get to make the good / bad decision.
		
		Right, so using biological reward, we have a function [X × Γ → Good, Bad]. Call the (X, \Gamma) an "experience". We sample experiences to get a learned version of this. 
		
		So what about more general preferences? And what about meta-preferences? How do you get the general preferences? Examples:
		 	"I want there to be more freedom"
			"this is the coding style I want to follow"
			"Poverty is bad"
			"We should be more environmental"
			"You should not steal things"
			
		Possibly all learned from different experiences. Can be changed by having more experiences. What about meta-preferences?
			"I want to not be gay" [because it would be easier, making these choices causes me pain]
			"I wish I liked broccoli" [because it would be easier, making these choices is not good for me]
			"I want to be an ethical person"
			"I wish I didn't feel anything"
			"I want to be more rational"
			"I want to prefer tofu to meat"
		
			==> Sometimes formed due to negative experiences associated with making choices according to other preferences. Sometimes formed with reasoning: this preference has these consequences, which are bad (conflict with my other preferences) and so I need to re-update my  
			
			Also common to have meta-preferences which align with preferences:
				"It's a good thing I like exercise"
				"I prefer to choose coffee over tea (for these reasons?) and I do"
				
			Using meta-preferences
				"I want to only want things that are possible"
		
		Also have preferences over other peoples' preferences:
			"I want people to love me"
			"I want more peole to "
			"I prefer if people wanted to learn"
			
			.. but these are actually kind of just like preferences over the environment.
			
			
		How do people's preferences change?
			- New experience changes the internal preference model to be closer to the biological one
			- 
	
Simluating preferences with just good / bad operators: 
	it's good that (I choose X over Y) and not good that (I choose Y over X)


Without a model of the world, it cannot be possible to ensure that your preferences are actually things that could be acted on. Emperically, as people discover things about the world, their preferences for impossible things go away (no more dreaming of discovinng unicorns or perpetual motion machines). 

Your preferences are OVER possible worlds, so it doesn't make sense not to have them change as your beliefs about the world and what is possible change. 

Need to explain: why not just wirehead? Why not just do a bunch of drugs that make you happy? Answer: this doesn't satisfy some preferences about the real world that we've already collected. By Hebbian mechanics, the original reward function has rubbed off on other things (self-actutalization, sport, etc.,) and now these carry real value --- so maximizing the original thing won't work anymore. 

----------------------

# Wishlist

PREVIOUS WISHLIST:
 - Death: how to incorporate deaths of agents into theory
 - Connections to runs/systems framework + epistemic logic
 - macro-micro / zooming effects of drawing agent boundaries in different places
 - actions as dependent types
 - reductions to well-known theories if you provide certain oracles
 - sub-agents
 - connections to other leraning theory: particuarly MAML and active learning
 - connections to bounded complexity 
 - connections to causal models
 - definitions of power / influence between agents


NEW IDEAS:
 - Formalization of Hebbian mechanics
 - Explanation of cognitive biases in this model + illustration that 

  
THEOREM WISHLIST:
 - 
	
	Intuition: 
