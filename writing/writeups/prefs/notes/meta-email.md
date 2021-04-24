@clarification Also it addresses an issue I have with many other approaches: you can assign high utility to configurations of variables that you know have no physical meaning. This feels so incredibly alien: it seems like you should be neutral in this situation. Here's my justification for this in terms of the previous characterization: if you believe A =


If there's more back and forth relevant to specific parts, I wonder if google docs, overleaf, or github  would be better. I'm happy to do any of these, including sticking with email.


-------- On Less is More --------
I was trying to follow the advice you gave me of putting down enough examples with enough properties that I can argue that my theory is the only one that can handle everything; some of these are just proactive protection against "why can't we just use this existing approach", including some which are intended as obvious sanity check desiderata, ruling out things like a separate, random utility function at each time, or obviously irrational behavior. Most are me are me outlining interesting properties I see as positive, that I'm pretty sure my theory has, just so I can reference the examples to explain it. The last one was a bookmark on something I think is very important but far in the future.

I know a paper wouldn't look like this but my intuition for the right number and flavor of examples has not been very good in the past and I figured I would over-correct and mitigate the risk of another week where I still needed to come up with informal examples. I'm trying to be better about communicating my thoughts rather than going off and doing math





-------- Thoughts on this Exercise, and Worries about what happens after I formalize the examples here --------
I think writing down the properties was immensely useful, and the toy examples were helpful insofar as I could use them to think more carefully about the properties I've written down, and also to remind me of properties that I had temporarily forgotten about. 

At the same time, I'm a bit frustrated that I'm thinking of all of these examples before proving the relevant theorems. I agree that things have to be grounded to be useful, but I have applications in mind, that are not exactly the kind of examples you want (as you've pointed out). I also acknowledge that examples are an invaluable communication device, but putting enough detail to ground them in specific instantiations of everything, before I have actually verified the relevant properties of my constructions, seems problematic because it:
 takes so much time and mental energy for me to do this, and also it's not really necessary that the examples be a static target we fit to; the examples are free to change as we discover things about the math which makes some more or less well-fit to explanation 
 exposes irrelevant features and draws attention to specifics that I don't really want to model just yet, and away from the contribution I want to make
 makes me sell things (these properties) that I'm not 100% sure that I'll be able to guarantee
makes it seem like the point is to deal with the examples--- but really, the point is either to explore the math and representations, guarantee properties, or to solve the problems presented by the application domain. It seems like a particular kind of over-fitting: not problematic because we'd then think of other examples to expose holes, but merely a sub-optimal use of time.

The bigger problem is that I might be wasting effort while I have all these looming concerns about other properties which might not hold, that I'm not sure I can capture with examples. Here are a few of the things I have in the back of my mind right now:
Does transitivity in B result in a transitive image on A, after filtered through a stochastic matrix A->B? If not, what conditions do we need to impose on this matrix? Do we want to replace links with something else?
How do we deal with the condorcet paradox? That is, different transitive images from different domains right now are naively combined into an intransitive preference; do we take matrix powers? Do we let this slide, and hope the other domains take care of it? Do we restrict to boolean variables so local transitivity is never a problem, and we just export the computation to the gradient descent, and hope that some global properties hold?
We need a kernel / distance metric in order to make the learning part reasonable. Should I put this into the formalism now, or just have gimped "learning" which doesn't generalize?
I still haven't worked out the details of conditional preferences. With some work, they should be ok on their own, but I worry about being able to define a reasonable kernel for them.
Will I be able to represent savage axioms 4-7? What kinds of contortions to the formalism would be required to make this happen?
Are there non-global, local minima to this optimization problem? Are they problematic?
What is the homotopy of this look like? Is it always contractable? If so, this undermines several of my examples, because I can't give a static case where forces counter-balance each other.
Dealing with these is in my mind much more important than dealing with the synthetic examples I derived from other properties I'm more certain of. And I worry that in the process of dealing with these, we'll have to throw away and re-compute most of our examples.

All of this said, I'm going to write down how these examples work formally; I'm certain that it will be helpful for me to think through them more carefully, but I worry that one of these other problems will render the whole thing irrelevant when I have to make substantial changes to my formalism.
