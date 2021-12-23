# Part 1: 3D multi-XPUs diffusion solver

The goal of this part is to implement the 3D diffusion equation:

![Formula 1](https://user-images.githubusercontent.com/50950798/145724451-f8f058b9-6265-4c5e-94e9-968e64ece804.png)

using the dual-time method where the physical time-derivative (dt) is defined as physical term and we use pseudo-time (&tau;) to iterate the solution:

![Screenshot from 2021-12-23 13-14-27](https://user-images.githubusercontent.com/50950798/147239208-08aa2b8d-9777-42b8-9419-cc202e8d5e94.png)


And we are also interested in a steady state solution:

![Screenshot from 2021-12-23 01-09-10](https://user-images.githubusercontent.com/50950798/147168780-b14ead68-8220-4eba-8087-9c68b6ea00d7.png)


In both cases will also make use of acceleration/damping to enforce scaling of the pseudo-transient iterations.


The next goal is to implement a version of this PDE solution that can run on multiple CPUs and/or GPUs, on a single Computer or even a distributed system. For this task we will make use of [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) which enables us to write code that can be deployed both on a CPU or a GPU (or short XPU). Also we will use [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) for distributed parallelization (using MPI) of the XPU solution.

Using the different implementations we will then perform some performance test and also do some scaling experiments.

## Methods

### (Pseudo) Temporal Discretation
As previously stated, the goal is to solve the linear diffusion equation in 3 Dimensions using the dual-time method, where the physical time-derivative (dt) is defined as physical term and we use pseudo-time (&tau;) to iterate the solution:

![Screenshot from 2021-12-14 19-03-27](https://user-images.githubusercontent.com/50950798/146054589-f514cdde-0ddd-46ae-bad6-225a8a9a0d71.png)

Which implies the following (pseudo-time) discretisation:

![Screenshot from 2021-12-14 19-05-28](https://user-images.githubusercontent.com/50950798/146054920-bf9d2cb9-8908-4563-918f-6399df4deb75.png)

For practical purposes we divide the solution into 3 steps. 
1. First we Calculate `H_Res`: 

![Screenshot from 2021-12-14 19-20-56](https://user-images.githubusercontent.com/50950798/146057084-3397ec74-af7c-4675-912d-b63aabee2241.png)

And in the case where we want to get the steady state solution, we calculate `H_Res` this way:

![Screenshot from 2021-12-23 01-13-04](https://user-images.githubusercontent.com/50950798/147168968-3077b8e4-a2fd-48c8-a9bc-f41f6c7637d9.png)


2. Since we use damping, we update `dH/dτ` in the seccond step with a damping parameter:

![Screenshot from 2021-12-14 19-23-02](https://user-images.githubusercontent.com/50950798/146057387-b265abab-26ec-46c4-b2b5-750ca502188c.png)


3. Finally we update `H`:

![Screenshot from 2021-12-14 19-27-20](https://user-images.githubusercontent.com/50950798/146057940-c09b0674-6270-409c-a027-495ddf221291.png)

<details>
<summary><strong>Sidenote on Numerical Differentiation</strong></summary>

We used the `FiniteDifferences3D` submodule from [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) for the numerical differentiation. Like the name suggests, this submodule provides numerical differentiation via the [Finite Differences Method](https://en.wikipedia.org/wiki/Finite_difference_method).

Here is an example of a pratial numerical derivation of `H` in direction `x`: 

![Screenshot from 2021-12-14 19-34-29](https://user-images.githubusercontent.com/50950798/146059076-227d5d32-f851-4baa-ad65-b606cf56fd44.png)

Equivalently here is a pratial numerical 2nd derivation of `H` in direction `x`: 

![Screenshot from 2021-12-14 19-38-17](https://user-images.githubusercontent.com/50950798/146059538-78735d8f-94ea-46f9-8d2b-ede067ac3c0b.png)

Using this submodule provied us with the following abstraction:

![Screenshot from 2021-12-14 19-40-25](https://user-images.githubusercontent.com/50950798/146059863-b992f72a-fb1a-4569-911e-65af77191158.png)

And thus the 2nd derivative of `H` in direction `x` can be written as:

![Screenshot from 2021-12-14 19-42-22](https://user-images.githubusercontent.com/50950798/146060105-ecb43eb9-826c-45ec-87fa-4b648fc7d3d5.png)
</details>


### Optimizations and Spatial Discretation
To optimize we will replace all divisions with inverse multiplications, since multiplication is much faster than division.
Also we will transform the equation so we get the least amount of operations possible. Ans precompute values where possible.
So lets look again at `ResH`:

![Screenshot from 2021-12-14 19-20-56](https://user-images.githubusercontent.com/50950798/146061372-06bb4755-25d3-4472-853c-0cb1320da68d.png)

And since we solve this Equation in 3 Dimentions, we get this equivalent formula:

![Screenshot from 2021-12-14 20-00-25](https://user-images.githubusercontent.com/50950798/146062548-6da42bab-78e9-461c-8d5c-6e2ec8308215.png)

Then we use the abstraction described in the Numerical Differentiation sidenote to discretize the PDE in the spatial domain:

![Screenshot from 2021-12-14 20-04-48](https://user-images.githubusercontent.com/50950798/146063107-84350216-dc72-4016-bd5e-1f974a2dd84e.png)

And now the optimizations described above are pretty straight forward:

![Screenshot from 2021-12-14 20-05-13](https://user-images.githubusercontent.com/50950798/146063167-b1161217-86cf-4622-9f72-21b774d03364.png)

Since we cant really optimize the other to steps, we now have the following three calculation steps in each iteration:

```julia 
# Step 1 (dual time)
@all(ResH) = -(@inn(H) - @inn(Hold)) * _dt + (@d2_xi(H)*D_dx² + @d2_yi(H)*D_dy² + @d2_zi(H)*D_dz²)

# Step 1 (steady state, only pseudo time)
@all(ResH) = @d2_xi(H)*D_dx² + @d2_yi(H)*D_dy² + @d2_zi(H)*D_dz²

# Step 2
@all(dHdt) = @all(ResH) + damp * @all(dHdt)

# Step 3
@inn(H) = @inn(H) + dτ * @all(dHdt)
```




### Solution approach

As the name "dual-time" suggests, we iterate over 2 types of times. The physical time and the pseudo time.
In this case the physical timestep and the total physical time is given by: `dt = 0.2` and `ttot = 1.0` (both in seconds).
So we iterate over the physical time and in each physical timestep `t` we iterate over the pseudo time &tau; until the L2-norm of the equation's residual `(norm(ResH)/sqrt(length(ResH))` is smaller then the absolute tolerance given by: `tol = 1e-8`. Then we increment `t` by `dt` and update H\^t with the value of H\^&tau; And if `t < ttot` we start the pseudo time loop again. Here is a code snippet that illustrates the dual time loop:

```julia 
#(...)
t = 0

# Physical time loop
while t<ttot

    it_τ = 0

    # Pseudo-transient iteration
    while err>tol && it_τ<itMax
        
        # Calculate ResH, dHdt and H
        #(...)
        
        # Calculate error
        #(...)

        it_τ += 1
    end

    # update physical time step
    t += dt

    #(...)
end
#(...)
```

And if we just want the steady state solution, we just need the pseudo-transient iteration loop:

```julia 
#(...)
it_τ = 0

# Pseudo-transient iteration
while err>tol && it_τ<itMax

    # Calculate ResH, dHdt and H
    #(...)

    # Calculate error
    #(...)

    it_τ += 1
end
#(...)
```

### Hardware

The hardware used to perform all simultaions and scaling experiments:
  - CPU: Intel® Core™ i7-11700 (8C16T, T_peak=50GB/s [[1]](#1))
  - GPU: NVIDIA GeForce RTX 3060 (T_peak=360 GB/s, 199 GFLOPS (FP64) [[2]](#2))

Except for the Multi GPU Experiment, where we used:
  - Multi-GPU Node: 4x NVIDIA GeForce GTX TITAN X (T_peak=480 GB/s, 209 GFLOPS (FP64) [[3]](#3))

## Results

**Dual Time Solution with grid size 512x512x256 - slice at z=128 - using 4 Titan X**

![3DDiffusion](img/diffusion3D_multixpu.gif)

Generated by executing: `julia -O3 --check-bounds=no -t 4 ./scripts-part1/diffusion3D_visualize.jl`.

### Performance
We are using the [performance metric](https://github.com/omlins/ParallelStencil.jl#performance-metric) proposed in the [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) library. (`T_eff = A_eff/t_it`)

In this case, the `A_eff` metric was calculated as follows:
```julia
reads = length(Hᵗ)                                     # Read Only Memory Access: Hᵗ
updates = length(H) + length(dHdt)                     # Update Memory access: H and dHdt
A_eff = 1e-9 * (2 * updates + reads) * sizeof(Float64) # Effective main memory access per iteration [GB]
```
We do not count `ResH` as it is a convenience array and thus could be skipped.
As you can see in the plots below, using the above mentioned hardware, we are definitely compute bound for this problem.

#### CPU Performance

An Intel® Core™ i7-11700 (8C16T, T_peak=50GB/s [[1]](#1)) was used for the CPU performance benchmark.
We executed the `diffusion3D_benchmark_cpu.jl` 4 times, with 1, 4, 8 and 16 threads respectively.

<p align="center">
  <img alt="Light" src="img/diffusion3D_scaling_experiment_cpu_1threads.png" width="49%">
&nbsp;
  <img alt="Dark" src="img/diffusion3D_scaling_experiment_cpu_4threads.png" width="49%">
</p>

<p align="center">
  <img alt="Light" src="img/diffusion3D_scaling_experiment_cpu_8threads.png" width="49%">
&nbsp;
  <img alt="Dark" src="img/diffusion3D_scaling_experiment_cpu_16threads.png" width="49%">
</p>

All of the 4 plots were generated by running `julia -O3 --check-bounds=no -t <num_threads> ./scripts-part1/diffusion3D_benchmark_cpu.jl`, where `<num_threads>` is replaced by the number of Threads.

We see here an interesting phenomenon with the 32 and especially the 64 grid size, when we increase the number of threads. The `T_eff` values are significantly higher compared to those with higher grid sizes. This behavior is probably the result of good caching from the CPU, since a grid size of 32 and 64 fit nicely in the Cache. The Cache size is about 16MB[[1]](#1), and we allocate 8 Arrays. So a grid size of 64: `8 * (64^3 * 4) ≈ 8MB` fits well in a 16MB cache. But a grid size of 128: `8 * 128^3 * 4 ≈ 67MB`, does of course not fit. So the true `T_eff` would be reflected by a grid size of 128 and up.


#### GPU Performance

An NVIDIA GeForce RTX 3060 (T_peak=360 GB/s, 199 GFLOPS (FP64) [[2]](#2)) was used for the GPU performance benchmark.
Using the `diffusion3D_benchmark_gpu.jl` script, also shows us the optimal local problem size, which we want to use later for the Multi GPU scaling experiment. In this case 256 looks like the optimal problem size.

![E5](img/diffusion3D_scaling_experiment_gpu.png)

Generated with `julia -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_gpu.jl`.


#### Weak scaling Experiment
Now that we have the optimal local problem size for a GPU (256), we will run a week scaling experiment using multiple GPUs. (4x NVIDIA GeForce GTX TITAN X)
To assess the performance when scaling to multiple GPUs we created two plots:

|**Number of GPUs vs Effective memory throughput**|**Number of GPUs vs Parallel efficiency <br> ([time using n GPUs]/[time using 1 GPU])**|
|---|---|
|![WS1](img/scaling_experiment_mgpu_teff.png)|![WS2](img/scaling_experiment_mgpu_pareff.png)|

The plots were generated by executing the `diffusion3D_benchmark_multigpu.jl` script 4 times:
- `~/.julia/bin/mpiexecjl -n 1 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl`
- `~/.julia/bin/mpiexecjl -n 2 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl`
- `~/.julia/bin/mpiexecjl -n 3 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl`
- `~/.julia/bin/mpiexecjl -n 4 julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_benchmark_multigpu.jl`

The NVIDIA GeForce GTX TITAN X and NVIDIA GeForce RTX 3060 GPUs are on paper pretty similar in terms of 64 bit FLOPs <br> (209 vs 199) and since the problem is compute bound, we expected about the same performance, which one can verify with the plots. (`T_eff ≈ 125 GB/s` for both GPUs, with a grid size of 256)


#### Work-precision diagrams

**Iterations to steady state vs Grid size**
<p align="center">
  <img alt="Light" src="img/diffusion3D_workprecision_1.png" width="49%">
&nbsp;
  <img alt="Dark" src="img/diffusion3D_workprecision_2.png" width="49%">
</p>

**Value at domain point (5,5,5) vs Grid size**
<p align="center">
  <img alt="Light" src="img/diffusion3D_workprecision_3.png" width="49%">
&nbsp;
  <img alt="Dark" src="img/diffusion3D_workprecision_4.png" width="49%">
</p>

**Tolerance vs Convergence Behaviour to well converged solution**
<p align="center">
  <img alt="Light" src="img/diffusion3D_workprecision_5.png" width="49%">
&nbsp;
  <img alt="Dark" src="img/diffusion3D_workprecision_6.png" width="49%">
</p>

We consider a tolerance of `tol=1e-24` as a well converged solution, which is still numerically stable.

The plots were generated by executing the `diffusion3D_work_precision.jl` script: `julia --project -O3 --check-bounds=no ./scripts-part1/diffusion3D_work_precision.jl`.




## References
<a id="1">[1]</a> https://ark.intel.com/content/www/us/en/ark/products/212279/intel-core-i711700-processor-16m-cache-up-to-4-90-ghz.html

<a id="2">[2]</a> https://www.techpowerup.com/gpu-specs/geforce-rtx-3060.c3682

<a id="3">[3]</a> https://www.techpowerup.com/gpu-specs/geforce-gtx-titan-x.c2632
