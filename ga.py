import numpy as np
from ypstruct import structure

def run(problem, params):
    

    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters  of the Genetic Algorithm    
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc # parents poopulation count
    nc = int(np.round(pc*npop/2)*2) # child poopulation count
    gamma = params.gamma # to use during crossover
    mu = params.mu # to use during mutation
    sigma = params.sigma # to use during mutation

    # Empty Individual Template
    individual = structure()
    individual.value = None
    individual.cost = None

    # Best Solution Ever Found
    bestsol = individual.deepcopy() # no effect of change in any will cause change in another as its works as a refrence
    bestsol.cost = np.inf

    # Initialize Population
    pop = individual.repeat(npop)
    for i in range(npop):
        pop[i].value = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].value)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of each Iterations
    bestcost = np.empty(maxit)
    
    # Evolution loop
    for it in range(maxit):

        popc = []
        for _ in range(nc//2):

            # Perform Random Selection
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]
            
            # Perform Crossover
            c1, c2 = crossover(p1, p2, gamma)

            # Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)

            # Evaluate First Offspring
            c1.cost = costfunc(c1.value)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.value)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)
        

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out

def crossover(p1, p2, gamma=0.1):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.value.shape)
    c1.value = alpha*p1.value + (1-alpha)*p2.value
    c2.value = alpha*p2.value + (1-alpha)*p1.value
    return c1, c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.value.shape) <= mu
    ind = np.argwhere(flag)
    y.value[ind] += sigma*np.random.randn(*ind.shape)
    return y

def apply_bound(x, varmin, varmax):
    x.value = np.maximum(x.value, varmin)
    x.value = np.minimum(x.value, varmax)
