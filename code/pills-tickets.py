import numpy as np
import random
from matplotlib import pyplot as plt
import math

def sample_score( ϵ, agent):
    last_pill = 0;
    tickets = 0
    while(True):
        last_pill += 1
        p = (1 - ϵ) ** ( math.log(1+last_pill))
        if random.random() > p:
            return tickets

        act = next(agent)
        if(act == 0):
            tickets += 1
        else:
            last_pill = 0


def agentk(k):
    while(True):
        for i in range(k-1):
            yield 0
        yield 1

def agentk_rand(k):
    while(True):
        if(random.random()*k > 1):
            yield 0
        else:
            yield 1



def avgscore(ϵ, α, N=5000):
    return sum(sample_score(ϵ, α) for i in range(N))/float(N);


avgscore(0.01, agentk_rand(1))

a1 = agentk(2);
next(a1)

X = list(range(1,20))
Y = [avgscore(0.005, agentk(x)) for x in X]
Y2 = [avgscore(0.005, agentk_rand(x)) for x in X]
plt.plot(X,Y,'bo-', X,Y2, 'ro-')

X = [1,2,3,4,5,7,10,15,20,30,50]
E = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
Z = [[avgscore(ϵ, agentk(x)) for x in X] for ϵ in E]

Z[0]
plt.matshow(Z, cmap='Blues');
