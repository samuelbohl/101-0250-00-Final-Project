using Printf, LinearAlgebra, Plots

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


@parallel function compute_ResH!(H, Hold, ResH, D, _dt, D_dx², D_dy², D_dz²)
    @all(ResH) = -(@inn(H) - @inn(Hold)) * _dt + (@d2_xi(H)*D_dx² + @d2_yi(H)*D_dy² + @d2_zi(H)*D_dz²)
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


@views function diffusion_3D(;grid=(32,32,32), is_experiment=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D     = 1.0                   # diffusion coefficient
    ttot  = 1.0                   # total simulation time
    dt    = 0.2                   # physical time step

    # Numerics
    nx, ny, nz = grid
    tol        = 1e-8             # tolerance
    itMax      = 1e4              # max number of iterations
    if is_experiment              # we want to shorten the number of max iterations based on grid size durch the scaling experiments
        itMax  = 1e4/(nx*4)
    end
    
    nout       = 10               # tol check

    # Derived numerics    
    dx, dy, dz = lx/nx, ly/ny, lz/nz             # cell sizes
    dτ         = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1 # iterative timestep
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    _dt = 1.0/dt
    D_dx² = D/dx^2
    D_dy² = D/dy^2
    D_dz² = D/dz^2


    # Array allocation
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = copy(dHdt)
    ResH  = @zeros(nx-2, ny-2, nz-2)

    # Initial condition
    H0    = Data.Array([2.0 * exp(-(xc[ix] - 0.5*lx)^2 -(yc[iy] - 0.5*ly)^2 -(zc[iz] - 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
    H     = copy(H0)
    H2    = copy(H0)
    Hᵗ  = copy(H0)


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
            @parallel compute_ResH!(H, Hᵗ, ResH, D, _dt, D_dx², D_dy², D_dz²)
            # Step 2
            @parallel compute_dHdt!(ResH, dHdt, dHdt2, damp)
            dHdt, dHdt2 = dHdt2, dHdt # Pointer Swap
            # Step 3
            @parallel compute_H!(H2, H, dHdt, dτ)
            H, H2 = H2, H #Pointer Swap
            
            # Calculate L2 norm
            if (iter % nout == 0)
                err = norm(ResH)/sqrt(length(ResH))
            end

            iter += 1
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

    @printf("Ttime steps = %d, nx = %d, iterations tot = %d \n", it, nx, it)

    if is_experiment
        return T_eff
    end
    return xc, Array(H)
end