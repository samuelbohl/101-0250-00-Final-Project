# Part 2: 3D XPU Elastic Waves Solver
Solving the 3D Navier-Cauchy equations for elastic waves on XPU using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl).

## Methods
As a starting point, we used `acoustic_2D_elast3.jl` from the final task of exercise 3 in [lecture 7](https://eth-vaw-glaciology.github.io/course-101-0250-00/lecture7/#towards_stokes_flow_i_acoustic_to_elastic). The file already implemented a 2D Navier-Cauchy wave equation. The 2D (and 3D) Navier-Cauchy equations are best described by the following set of equations taken from the lecture:

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;P}{\partial&space;t}&space;=&space;-K\nabla_kv_k&space;" title="\frac{\partial P}{\partial t} = -K\nabla_kv_k " />

<img src="https://latex.codecogs.com/svg.image?\frac{\partial\tau}{\partial&space;t}=\mu\left(\nabla_iv_j&plus;\nabla_jv_i-\frac{1}{3}\delta_{ij}\nabla_kv_k\right)" title="\frac{\partial\tau}{\partial t}=\mu\left(\nabla_iv_j+\nabla_jv_i-\frac{1}{3}\delta_{ij}\nabla_kv_k\right)" />

<img src="https://latex.codecogs.com/svg.image?\rho\frac{\partial&space;v_i}{\partial&space;t}=\nabla_j(\tau_{ij}-P\delta_{ij})" title="\rho\frac{\partial v_i}{\partial t}=\nabla_j(\tau_{ij}-P\delta_{ij})" />

To add a 3rd dimension, we added 1 new normal-stress component (τzz) and 2 new shear-stress components (τyz, τzx):
```julia
@all(τzz) = @all(τzz) + dt*(2.0*μ*(@d_zi(Vz)/dz) - 1.0/3.0*@inn_xy(∇V))
@all(τyz) = @all(τyz) + dt*μ*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)
@all(τzx) = @all(τzx) + dt*μ*(@d_xi(Vz)/dx + @d_zi(Vx)/dz)

```
All the velocity calculations also had to be updated to incorporate the new components of the stress tensor:
``` julia
@inn(Vx) = @inn(Vx) - dt/ρ*(@d_xi(P)/dx - @d_xa(τxx)/dx - @d_ya(τxy)/dy - @d_za(τzx)/dz)
@inn(Vy) = @inn(Vy) - dt/ρ*(@d_yi(P)/dy - @d_ya(τyy)/dy - @d_za(τyz)/dz - @d_xa(τxy)/dx)
@inn(Vz) = @inn(Vz) - dt/ρ*(@d_zi(P)/dz - @d_za(τzz)/dz - @d_xa(τzx)/dx - @d_ya(τyz)/dy)
```
The only thing left then is to update `∇V`:
```julia
@all(∇V) = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
```

### Performance optimizations

After the solver was up and running, we noticed some things that could be optimized.

Temporary arrays that are used only once (`dVxdt`, `dVydt`, `dVzdt`, `dPdt`) were removed to reduce memory footprint and memory access time, while at the same time not increasing the amount of computational work.

`∇V`, although being a temporary array, was not inlined because doing so would significantly increase the computational work required, and we are likely compute-bound rather than memory-bound for this problem.

Constant divisions were also replaced by multiplication of their inverse, and consecutive multiplications were merged to reduce the computational work.

### File structure

The relavant files for this part are listed in the tree below.

```
.
├── docs
│   ├── img                                 # Automatically-generated plots and animations are saved here
│   └── part2.md                            # You are here :)
├── scripts-part2
│   ├── elastic_wave_3D.jl                  # Main file, contains the solver code and physical parameters
│   ├── elastic_wave_3D_benchmark_cpu.jl    # CPU performance benchmark (calls the main file)
│   ├── elastic_wave_3D_benchmark_gpu.jl    # GPU performance benchmark (calls the main file)
│   ├── elastic_wave_3D_testing.jl          # File used for reference testing (calls the main file)
│   └── elastic_wave_3D_visualize.jl        # Render the simulation (calls the main file)
└── test
    ├── part2.jl                            # Reference tests and unit tests
    ├── reftest-files
    │   └── test_2.bson
    └── runtests.jl
```
All commands to replicate the results below should be run from the repository's top-level directory.

## Results

For visualization, we took a 2D slice of the 3D pressure matrix `P` at `z=Lz/2`.

![3D elastic wave simulation](img/elastic_wave_3D.gif)

A resolution of 128x128x128 was used for the visualization. The physical properites used were `ρ=1.0`, `μ=1.0`, and `K=1.0`. To replicate:

```
julia --project -O3 --check-bounds=no scripts-part2/elastic_wave_3D_visualize.jl
```

### Performance

#### Performance metric

We are using the [performance metric](https://github.com/omlins/ParallelStencil.jl#performance-metric) proposed in the [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) library.

In our case, the `A_eff` metric was calculated as follows:
```julia
# Effective main memory access per iteration [GB]
A_eff = 1e-9 * sizeof(Float64) * (
    2 * (length(τxx) + length(τyy) + length(τzz) + length(τxy) + length(τyz) + length(τzx))
  + 2 * (length(Vx)  + length(Vy)  + length(Vz))
  + 2 * (length(P))
)
```
`∇V` isn't included in the calculation as it is only used as a temporary.

#### CPU Performance

An AMD Ryzen™ 5 5600G (6C12T, T_peak=47.68 GB/s [[1]](#1)) was used for the CPU performance benchmark.

![3D elastic wave CPU benchmark](img/elastic_wave_3D_scaling_experiment_cpu_6threads.png)

It is obvious from the graph that we are compute-bound for this problem on this particular CPU.

To run the CPU performance benchmark:
```
export JULIA_NUM_THREADS=<num_threads>
julia --project -O3 --check-bounds=no scripts-part2/elastic_wave_3D_benchmark_cpu.jl
```
Replace `<num_threads>` with the amount of physical cores in your CPU.

#### GPU Performance

An NVIDIA GeForce RTX™ 3060 (T_peak=360 GB/s, 199 GFLOPS (FP64) [[2]](#2)) was used for the GPU performance benchmark.

![3D elastic wave GPU benchmark](img/elastic_wave_3D_scaling_experiment_gpu.png)

A resolution of `512x512x512` wasn't tested because it didn't fit in the 12 GB of VRAM that the 3060 had.

In this case, too, we are bound by the computing power rather than the memory bandwidth.

To run the GPU performance benchmark:

```
julia --project -O3 --check-bounds=no scripts-part2/elastic_wave_3D_benchmark_gpu.jl
```

## References

<a id="1">[1]</a> https://en.wikichip.org/wiki/amd/ryzen_5/5600g

<a id="2">[2]</a> https://www.techpowerup.com/gpu-specs/geforce-rtx-3060.c3682
