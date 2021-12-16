using Printf, LinearAlgebra, Plots
using ImplicitGlobalGrid
import MPI

# IF GPU: ~/.julia/bin/mpiexecjl -n 1 julia --project ./scripts-part1/diffusion3D_dual_steady_multixpu.jl 
# IF CPU: ~/.julia/bin/mpiexecjl -n 12 julia --project ./scripts-part1/diffusion3D_dual_steady_multixpu.jl 
if !@isdefined USE_GPU
    const USE_GPU = false
end

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
#ParallelStencil.@reset_parallel_stencil()
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@views inn(A) = A[2:end-1,2:end-1,2:end-1]

@parallel function compute_ResH!(H, Hᵗ, ResH, _dt, D_dx², D_dy², D_dz²)
    @all(ResH) = -(@inn(H) - @inn(Hᵗ)) * _dt + (@d2_xi(H)*D_dx² + @d2_yi(H)*D_dy² + @d2_zi(H)*D_dz²)
    return
end

@parallel function compute_dHdt!(ResH, dHdt, dHdt2, damp)
    @all(dHdt2) = @all(ResH) + damp * @all(dHdt)
    return
end

@parallel function compute_H!(H2, H, dHdt, dτ)
    @inn(H2) = @inn(H) + dτ * @all(dHdt)
    return
end

norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))


@views function diffusion_3D(;grid=(32,32,32), is_experiment=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D     = 1.0                   # diffusion coefficient
    ttot  = 1.0                   # total simulation time
    dt    = 0.2                   # physical time step
    # Numerics
    nx, ny, nz = grid
    tol        = 1e-8             # tolerance
    itMax      = 1e5              # max number of iterations
    nout       = 10               # tol check
    # Derived numerics    
    me, dims, nprocs = init_global_grid(nx, ny, nz)
    dx         = lx/(nx_g());                              # Space step in x-dimension
    dy         = ly/(ny_g());                              # Space step in y-dimension
    dz         = lz/(nz_g());                              # Space step in z-dimension
    dτ         = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1            # iterative timestep
    _dt = 1.0/dt
    D_dx² = D/dx^2
    D_dy² = D/dy^2
    D_dz² = D/dz^2

    # Array allocation
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = copy(dHdt)
    ResH  = @zeros(nx-2, ny-2, nz-2)
    # Initial condition
    H     = @zeros(nx, ny, nz)
    H0    = Data.Array([2.0 * exp(-(dx/2 + x_g(ix,dx,H)- 0.5*lx)^2 -(dy/2 + y_g(iy,dy,H) - 0.5*ly)^2 -(dz/2 + z_g(iz,dz,H)- 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
    H     = copy(H0)
    H2    = copy(H)
    Hᵗ    = copy(H)
    
    t_tic = 0.0;t = 0.0; it = 0; ittot = 0

    damp  = 1-29/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Physical time loop
    while t<ttot
        iter = 0
        err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax

            if (iter == 11 && ittot == 0) t_tic = Base.time(); end

            # Step 1
            @parallel compute_ResH!(H, Hᵗ, ResH, _dt, D_dx², D_dy², D_dz²)
            # Step 2
            @parallel compute_dHdt!(ResH, dHdt, dHdt2, damp)
            dHdt, dHdt2 = dHdt2, dHdt # Pointer Swap
            # Step 3
            @parallel compute_H!(H2, H, dHdt, dτ)
            update_halo!(H2)
            H, H2 = H2, H # Pointer Swap
            
            # Vizualise and calc error
            iter += 1
            if iter % nout == 0
                err = norm_g(ResH)/sqrt(length(ResH))
            end
        end
        ittot += iter; it += 1; t += dt
        Hᵗ .= H
        if isnan(err) error("NaN") end
    end

    t_toc = Base.time() - t_tic
    # Performance
    A_eff = 9/1e9*nx*ny*nz*sizeof(Float64);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    t_it  = t_toc/(ittot-10);                    # Execution time per iteration [s]
    T_eff = A_eff/t_it;                          # Effective memory throughput [GB/s]
    println("time(sec)=$t_toc T_eff=$T_eff")

    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=3), it, nx, ittot)

    # Postprocessing 
    finalize_global_grid()

    if is_experiment
        return T_eff
    end
    return xc, Array(H)
end