# Main points.

## Recall Qualitative vs Quantitative BN


## In Expectation vs Everywhere
α=0: Observation --- on average, this is the marginal
α=1: Structural Necessity --- regardless of context, this is the marginal.
    > Intuitively has to do with causality because interventions do not affect.
    > Also because the 
    
    

## Connection between Causality and MRFs
The inference algorithms (message passing, Gibbs sampling, etc.,) all make updates assuming it happens everywhere





# The email and my responses

I'm afraid this email reinforces my concerns about how far we are from 
converging, Oliver.   Don't respond to it.  We can talk tomorrow.

On 5/17/20 6:15 PM, Oliver Richardson wrote:
> Hi Joe,
>
> Thanks for your feedback.
> 4pm tomorrow works for me.
>
> *Definition 0 (your text + addition).*
>
>     /an edge with \alpha is that a label of alpha on an edge L from X
>     to Y represents your degree
>     of belief that X is causally independent of all variables other than
>     those in Y, and Y is minimal with respect to this (so that X is not
>     causally independent of the variables in Y)/
>
> /... and also any randomness in Y can be attributed to an independent 
> random variable local to the edge  L./
Your addition is unnecessary.  It follows from what I said
>
> To clarify, I propose the following one-sentence definition, which I 
> claim is equivalent:
It may be equivalent, but I don't understand it.
>
>
> *Definition 1.*
> /\alpha on an edge L : X -> Y represents the agent's degree of belief 
> that there exists a function f_L  such that  Y = f_L(X, N_L)  is 
> always true, /

I'm guessing that you're adding the N_L to convert Y=f_L(X) fro a 
probabilistic equation to a deterministic equation.  While I have no 
intrinsic problem with that, there is no N_L in sight in PDG, and I 
don't think it's a good idea to make up new variables that aren't there.
 *It's a noise variable corresponding to the link; in some sense it was given by the disigner.*

> /where N_L is a noise variable unique to L (i..e, no other such 
> function f_{L'} can depend on N_L for L' != L) , /

Why should we care whether anything else depends on N_L?  It certainly 
shouldn't be a problem if other variables depend on X.

> /and N_L is rendered causally and observationally independent of every 
> other variable by an intervention on Y/
independent of what?
> / --- and moreover that X is minimal (in the sense there is no 
> variable X' that is a deterministic, non-invertible function of X, and 
> edge L' : X' -> Y for which this is equally true)./
>
>
I don't understand this.  As near as I can tell, it has nothing to do 
with the notion of minimality that I had in mind.  
 *This is a stronger version of exactly yours; the value of a subset is determined by a projection out of it* 
To the extent that I 
understand, it seems wrong.  Are you saying that if X represents weight 
in pounds that there can't be an X' that represents weight in kilo?
 *This would be a deterministic, _invertible_ function of X*
 Why not?  Why are you putting this restriction on the PDG, and why should 
this have any bearing on the intuition for \alpha?
 *It's not a restriction on PDGs, but the definition for α.*

> Definition 1 is my preferred one, as it handles logical relationships 
> in a satisfying way, is the cleanest mathematically, places focus 
> back  on the edges, and makes the information profile a useful visual 
> tool for analysis. If you are unwilling to buy this, I might be 
> willing to settle for a weaker version which enforces \sum_\alpha <= 1 
> by definition.

As I said, I don't understand this and to the extent that I do, I 
certainly don't buy it.  I claimed that you could will have \sum \alpha  > 1
if you had logically related variables, 
 *This is true. For Defn1, sum\alpha>1*
 *For defn2, we set\alpha<1 for logically related variables, which is less satisfying.  (in fact, if you know the causal equation for Y to depend on X, then a link Z×Y → Y will have α=0, as it's not the causal equation.) But in this case it's not in confict because the trade-off is in the definition of \alpha. They just have to sum to 1.*
and I don't see how you can stop this.
>
>
> *Definition 2.*
> alpha_L for L : X -> Y is  the degree to which an agent believes L 
> (plus possibly some not-modeled noise variables) represents the causal 
> equation f_Y.  so that the \alpha's for all links into a node Y form a 
> distribution over { X_i } U { None of these }.
>
The requirement that that the \alpha's for all links into a node Y form 
a distribution over { X_i } U { None of these } is incompatible with the 
requirement that \alpha represents your degree of belief that f_L 
represents the causal equation.  
 *Why? There is only one causal equation for Y, f_Y. Suppose we have to attach it to a link L (or none). Which one do we choose? The distribution over choices is this notion of α.* 
So I simply don't understand this.
>
> A few inline comments:
>
>     Note that we have never required (or even
>     discussed) logical independence.  I do so in the attached (still
>     unpublished) paper, which you may want to look at. Without logical
>     independence, I see no reason that we can't have the sum of the
>     \alpha's
>     being greater than 1 with the current intuition.   If you want to
>     prevent this, then you *must* have a different intuition, and I'd
>     like
>     to see that expressed clearly (in one sentence, as I said above).
>
>
> I will take a closer look at this paper before our meeting.
>
> However, I believe that the definition I gave actually implicitly 
> captures the property you're looking for, because of the strengthening 
> of "subset" to "deterministic non-invertable function", combined with 
> the robustness to intervention.
>
>     - Another consequence of this definition, as I've observed before
>     is if
>     Y = X + random noise, where X is either 0 or 1, and the noise is
>     chosen
>     between 1000 and 2000, an we know this, then we should assign the
>     edge
>     from X to Y weight 1 (assuming that there is no variable representing
>     random noise).  However, the value of information from learning X is
>     close to 0.  This shows that, whatever \alpha is measuring, it has
>     next
>     to nothing to do with value of information.    Again, if you disagree
>     with this, you must have a different intuition for X, and I'd like to
>     see it expressed clearly, in one sentence.
>
>
> I like this example. We should indeed set \alpha =1 if we know this.
> To the extent the agent is unclear if X makes an impact, alpha may be 
> split between this one and an edge 1 -> Y.
>
> Value of information is still relevant. Here are three examples:
> (1) This tells us that we should still pay full price for Y even if we 
> know a third variable Z,
> (2) You should still pay (a very tiny) amount less both X and Y 
> together, and
> (3) The joint information can still be very valuable. If you discover 
> that Y = 1000, or Y = 2001, respectively, you know for sure that X=0 
> or X=1.
>

Value of information may be relevant to lots of things, but it's 
irrelevant to \alpha.  That is, it should have no impact on \alpha at 
all (except perhaps if \alpha > 0 the value of information might be > 0.
> Keep in mind that the \alpha's only control how good or bad it is to 
> share information with variables qualitatively, based on the graph.
The alpha's do not control anything!  They simply represent a belief.  I 
have no idea what it means "to share information with variables".  You 
cannot start using language like that out of the blue.

> They do not suggest any concrete value of how much is appropriate.
> The assertions about the actual numbers are given by the \betas.
>
I don't understand this at all.
>
>     Whether you agree with this or not, you seem to be moving to a world
>     where what you are interested in is not just observational
>     probabilities, counterfactual probabilities.  I am perfectly
>     comfortable
>     with that, but I would note that it's not in the spirit of our
>     current
>     semantics, which talks about
>     observational probabilities. 
>
>
> Recall that with this definition, holding causally => holding 
> observationally.

Only if \alpha = 1.
 *α<1 does not change the meaning of the cpt, just your certainty that it holds causally in addition to observationally*
> By default including a cpt is an assertion that it holds observationally,
Where did this default come from, and where is it represented?
 *By "default", I just mean before we start thinking about α's, i.e., the first semenatcs / γ=0  or with α=0*

> and asserting \alpha=1 elevates this to a causal assertion.

But in general we won't have \alpha=1, which means that we have to 
represent both causal information and observational information.
 *OK but we store it in such a way that the cpts would always be the same*
>
> It is true that the counterfactual probabilities obtained in this way 
> are only a small subset of them, but because they can specify the 
> whole causal model, many of the rest can be generated by composition.
I don't understand that.  (But don't answer it.)
>
>     This leads to an obvious question: can
>     we usefully think of a PDG as representing a set of counterfactual
>     probabilities. That is, I'm looking for an analogue of our current
>     first
>     semantics, where we replace observational probabilities by
>     counterfactual probabilities (which are more general).  If we
>     bring in
>     counterfactual probabilities (which, as I said, I'm perfectly
>     comfortable doing), we can't bring them in out of the blue only in
>     the
>     discussion of \alpha.  They should be there all along.
>
>     So where does this leave us?  I think that there is a nice clean
>     paper
>     where you can present unweighted PDGs and have up to Section 4.1
>     in the
>     current writeup now on overleaf.  You don't need alpha or beta for
>     that
>     writeup.  You can then write a one-page conclusion about where
>     else that
>     can go and submit it to NeurIPS.  For that paper, you don't have to
>     worry about explaining alpha and beta.  I have no problem if you
>     add a
>     conclusion in that paper saying that you want ultimately to consider
>     weighted PDGs (with some intuition for \alpha, perhaps that above),
>     perhaps with their semantics being counterfactual probabilities,
>     and say
>     that, with a more general scoring function, they can capture
>     exponential
>     families. 
>
>
> I should mention that the current formulation exactly captures MRFs 
> and their associated exponential families, when  α = β / γ  for every 
> edge.
Two big issues.  Can we explain why the intuition behind \alpha let's us 
explain exponential distributions?  
*This  definition of α is stronger than we need. The thinking causally and allowing for interventions is in some sense the "right" way to set α, but all we relaly need is a robustness of some kind --- here it just needs to be to sampling, not adversarial interventions.*
*Message passing and Gibbs sampling can be thought of as exploting this causality: they make use of the cpts holding counter-factually.*
 
We have to connect the intuition for 
\alpha to the technical results.  Since MRF's are not defined in terms 
of causality and causal probabilities, this will be nontrivial.  
 *That's why it was non-trivial to get this definition, but I think the rest is straightforward.*
And while I recall that you had this other term \gamma., it too will need to 
be motivated explained.

>
> A consequence of this is that if we fix every \alpha = \beta = 1, then 
> the third semantics prescribes the same distribution as the product of 
> factors in the factor graph. As I have tried to explain in the past, 
> this is not ideal; one reason is that the resulting distribution will 
> not even be consistent with the cpts (even though it is the product of 
> them). The mathematical reason for this is that the sum of \alphas 
> into a node exceeds 1.
Perhaps needless to say, I don't understand this at all.  You're making 
it sounds like a problem that the sum of the alpha's exceeds 1.  
According to my intuition, which you have agreed with, this can happen 
only if you have logical relations between the variables.  Is that the 
case here?  If not, the way you're thinking of alpha in this context is 
different from the intution that you've agreed to. This is a problem.  
You have to give an intuition for alpha that matches its usage.
>
>       I know that this is not the paper that you hoped to write,
>     but (a) as I said, I think that if you want to bring in
>     counterfactual
>     probabilities, you need to rethink the whole presentation and
>     story in
>     any case and (b) whereas the paper that I propose is close to done
>     and
>     tells a reasonable story, I can't imagine that we could converge on a
>     paper anywhere close to what you'd like to do by the NeurIPS
>     deadline.
>     And even if we did, I suspect that we'd want that to be the second
>     paper, not the first.
>
>
> I agree with this analysis, and in particular that saying too much 
> about causality here would be too much.
>
> However, I'm still nervous about this approach. At this point, I now 
> have a really pretty, nearly complete picture of how strict PDGs fit 
> in with and generalize other graphical models. Though it is not 
> obvious that this picture has anything to do with interventions, 
> it nonetheless hinges crucially on the existence of the parameter \alpha.
As I've said above, if the way that you use the alpha to paint this 
picture has nothing to do with interventions, then the intuition that 
you've been struggling to explain to me is the wrong one.

*It has a lot to do with interventions; it's just that...*
    1. we must be more careful describing α for a PDG, as the modularity from graph unions, etc., allows us to do things like perform interventions.
    2. It's not usually couched in this language.
