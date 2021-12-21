# Part 2: 3D XPU Elastic Waves Solver
Solving the 3D Navier-Cauchy equations for elastic waves on XPU using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl).

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;P}{\partial&space;t}&space;=&space;-K\nabla_kv_k&space;" title="\frac{\partial P}{\partial t} = -K\nabla_kv_k " />

<img src="https://latex.codecogs.com/svg.image?\frac{\partial\tau}{\partial&space;t}=\mu\left(\nabla_iv_j&plus;\nabla_jv_i-\frac{1}{3}\delta_{ij}\nabla_kv_k\right)" title="\frac{\partial\tau}{\partial t}=\mu\left(\nabla_iv_j+\nabla_jv_i-\frac{1}{3}\delta_{ij}\nabla_kv_k\right)" />

<img src="https://latex.codecogs.com/svg.image?\rho\frac{\partial&space;v_i}{\partial&space;t}=\nabla_j(\tau_{ij}-P\delta_{ij})" title="\rho\frac{\partial v_i}{\partial t}=\nabla_j(\tau_{ij}-P\delta_{ij})" />

## Methods
As a starting point, we used `acoustic_2D_elast3.jl` from the final task of exercise 3 in [lecture 7](https://eth-vaw-glaciology.github.io/course-101-0250-00/lecture7/#towards_stokes_flow_i_acoustic_to_elastic).

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
For visualization, we took a 2D slice of the 3D pressure matrix `P` at `z=Lz/2`.

## Results

### The physics you are resolving

### Performance

#### Memory throughput

#### Weak scaling

#### Work-precision diagrams

## Discussion

## References
