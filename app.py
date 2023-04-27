import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure #modified version of dictionary
import ga

# Cost Function to optimize by making it minimum
def sphere(x):
    return sum(x**2)

# Problem Definition
problem = structure()
problem.costfunc = sphere
problem.nvar = 5 #NEED TO UNDERSTAND WHAT IS THIS
problem.varmin = -10
problem.varmax =  10

# GA Parameters
params = structure()
params.maxit = 100
params.npop = 1000
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
out = ga.run(problem, params)

# Results
plt.plot(out.bestcost)
#plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()