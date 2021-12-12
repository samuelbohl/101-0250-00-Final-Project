# Part 1: 3D multi-XPUs diffusion solver
Steady state solution of a diffusive process for given physical time steps using the pseudo-transient acceleration (using the so-called "dual-time" method).


## Intro
As the title suggests the goal of this part is to implement the 3D diffusion equation:

![Formula 1](https://user-images.githubusercontent.com/50950798/145724451-f8f058b9-6265-4c5e-94e9-968e64ece804.png)

using a dual-time method where the physical time-derivative (dt) is defined as physical term and we use pseudo-time (&tau;) to iterate the solution:

![Fromula 2](https://user-images.githubusercontent.com/50950798/145724843-911afb30-9c84-41cf-9a4b-10f09e56dcd3.png)

We will also make use of acceleration/damping to enforce scaling of the pseudo-transient iterations, finding the optimal damping term in the process.

The next goal is to implement a version of this PDE solution that can run on multiple CPUs and/or GPUs, on a single Computer or even a distributed system. For this task we will make use of [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) which enables us to write code that can be deployed both on a CPU or a GPU (or short XPU). Also we will use [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) for distributed parallelization (using MPI) of the XPU solution.

Using the different implementations we will then perform some performance test and also do some weak scaling experiments.

## Methods
The methods to be used:
- spatial and temporal discretisation
- solution approach
- hardware
- ...

## Results
Results section

### 3D diffusion
Report an animation of the 3D solution here and provide and concise description of the results. _Unleash your creativity to enhance the visual output._

### Performance
Briefly elaborate on performance measurement and assess whether you are compute or memory bound for the given physics on the targeted hardware.

#### Memory throughput
Strong-scaling on CPU and GPU -> optimal "local" problem sizes.

#### Weak scaling
Multi-GPU weak scaling

#### Work-precision diagrams
Provide a figure depicting convergence upon grid refinement; report the evolution of a value from the quantity you are diffusing for a specific location in the domain as function of numerical grid resolution. Potentially compare against analytical solution.

Provide a figure reporting on the solution behaviour as function of the solver's tolerance. Report the relative error versus a well-converged problem for various tolerance-levels. 

## Discussion
Discuss and conclude on your results

## References
Provide here refs if needed.
