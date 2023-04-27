# Genetic Algorithm Optimization for Sphere Function

This project is a Python implementation of a Genetic Algorithm (GA) to optimize the Sphere function. The goal is to find the minimum value of the Sphere function by using GA optimization. The code is written in Python and uses the following libraries:

- NumPy: for numerical computation
- Matplotlib: for visualization of results
- Ypstruct: for defining structures (modified version of dictionary)

## Problem Definition

The Sphere function is a simple optimization problem where the goal is to minimize the sum of squares of n variables.

In this project, we will use a 5-dimensional Sphere function defined as follows:

```python
def sphere(x):
    return sum(x**2)
```

## GA Parameters

The following parameters are used for GA optimization:

- `maxit`: maximum number of iterations
- `npop`: population size
- `beta`: selection pressure
- `pc`: parents population count
- `gamma`: crossover parameter
- `mu`: mutation probability
- `sigma`: mutation step size

These parameters are defined in the `params` structure as follows:

```python
params = structure()
params.maxit = 100
params.npop = 1000
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1
```

## Running the GA

To run the GA optimization for the Sphere function, we define the problem as a structure with the cost function and the number of variables. We also define the lower and upper bounds of the variables.

```python
problem = structure()
problem.costfunc = sphere
problem.nvar = 5
problem.varmin = -10
problem.varmax =  10
```

We then call the `run` function from the `ga` module with the problem and parameters as inputs.

```python
out = ga.run(problem, params)
```

## Results

The best cost of each iteration is stored in the `bestcost` array. We can plot this array to see how the GA optimization converges to the minimum value.

```python
plt.plot(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
```

The plot shows the convergence of the GA optimization to the minimum value of the Sphere function.

Note: The implementation of the GA algorithm is defined in the `ga` module, which includes the `crossover`, `mutate`, and `apply_bound` functions.
