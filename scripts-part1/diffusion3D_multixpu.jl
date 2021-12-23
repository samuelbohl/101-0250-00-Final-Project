using Printf, LinearAlgebra, Plots, ImplicitGlobalGrid, ParallelStencil, ParallelStencil.FiniteDifferences3D
import MPI

# Initialize Globals
if !@isdefined USE_GPU
    const USE_GPU = true
end

if !@isdefined BENCHMARK
    const BENCHMARK = false
end

if !@isdefined VISUALIZE
    const VISUALIZE = true
end

# Initialize `ParallelStencil.jl`
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

# Helper for shorter notation
@views inn(A) = A[2:end-1,2:end-1,2:end-1]

# Helper - Calculates global L2 norm using MPI
norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))

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

"""
    diffusion_3D(res)

Runs the 3D dualtime diffusion simulation on multiple processes using MPI via `ImplicitGlobalGrid.jl`.

# Arguments
- `res::Int`: xyz-resolutions of the simulation

# Returns (Depending on globals)
- If the BENCHMARK is true ->
    `T_eff`, `t_toc`, `nprocs` and `me`: Effective memory throughput [GB/s], Total calculation time, number of processes and process id
- If BENCHMARK is false ->
    `xc` and `H`: The global x-coordinate Array and the global solution array
"""
@views function diffusion_3D(res::Int)

    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D     = 1.0                   # diffusion coefficient
    ttot  = 1.0                   # total simulation time
    dt    = 0.2                   # physical time step

    # Numerics
    nx, ny, nz = res, res, res
    tol        = 1e-8             # tolerance
    itMax      = 1e5              # max number of iterations
    nout       = 80               # tol check

    # Initialize MPI
    me, dims, nprocs = init_global_grid(nx, ny, nz)
    @static if USE_GPU
        select_device()
    end

    # Derived numerics    
    dx         = lx/(nx_g())                    # Space step in x-dimension
    dy         = ly/(ny_g())                    # Space step in y-dimension
    dz         = lz/(nz_g())                    # Space step in z-dimension
    dτ         = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1 # iterative timestep
    damp       = 1-29/nx                        # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    _dt = 1.0/dt
    D_dx² = D/dx^2
    D_dy² = D/dy^2
    D_dz² = D/dz^2

    # Array allocation
    H     = @zeros(nx  , ny  , nz  )
    H2    = @zeros(nx  , ny  , nz  )
    Hᵗ    = @zeros(nx  , ny  , nz  )
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = @zeros(nx-2, ny-2, nz-2)
    ResH  = @zeros(nx-2, ny-2, nz-2)

    # Initial condition
    H  .= Data.Array([2.0 * exp(-(dx/2 + x_g(ix,dx,H)- 0.5*lx)^2 -(dy/2 + y_g(iy,dy,H) - 0.5*ly)^2 -(dz/2 + z_g(iz,dz,H)- 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
    H2 .= H
    Hᵗ .= H

    # Preparation of visualisation
    @static if VISUALIZE
        gr()
        ENV["GKSwstype"]="nul"
        anim = Animation()
        xc   = LinRange(dx/2, lx-dx/2, nx_g()) 
        yc   = LinRange(dy/2, ly-dy/2, ny_g())
        init_extrema = extrema(Array(H))
        nx_v = (nx-2)*dims[1]
        ny_v = (ny-2)*dims[2]
        nz_v = (nz-2)*dims[3]
        H_v  = zeros(nx_v, ny_v, nz_v)
        H_nohalo = zeros(nx-2, ny-2, nz-2)
    end

    # Benchmark variables
    @static if BENCHMARK
        warmup = 10         # Number of warmup-iterations
        t_tic  = 0.0        # Holds start time
        itMax  = 1e4/nx     # scale max number of iterations down my grid size
    end
    
    # initial physical time loop conditions
    t_tic = 0.0
    t = 0.0
    it_t = 0
    ittot = 0

    
    # Physical time loop
    while t<ttot

        # initial pseudo-transient loop conditions
        it_τ = 0
        err = 2*tol

        # Pseudo-transient iteration
        while err>tol && it_τ<itMax

            # Step 1
            @parallel compute_ResH!(H, Hᵗ, ResH, _dt, D_dx², D_dy², D_dz²)
            # Step 2
            @parallel compute_dHdt!(ResH, dHdt, dHdt2, damp)
            dHdt, dHdt2 = dHdt2, dHdt # Pointer Swap
            # Step 3
            @parallel compute_H!(H2, H, dHdt, dτ)
            update_halo!(H2) # update halo
            H, H2 = H2, H    # Pointer Swap
            
            # Calculate L2 norm (error)
            if it_τ % nout == 0
                err = norm_g(ResH)/sqrt(length(ResH))
            end

            # Start the clock after warmup-iterations
            @static if BENCHMARK
                if (it_t == 0 && it_τ == warmup+1)
                    t_tic = Base.time();
                end
            end

            # Render a slice of the 3D diffusion-map and save as an animation frame
            @static if VISUALIZE
                if (it_τ % nout == 0)
                    H_nohalo .= inn(H)     # Copy data to CPU removing the halo
                    gather!(H_nohalo, H_v) # Gather data on process 0 

                    # Visualize it on process 0.                              
                    if (me == 0)
                        opts = (aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), clims=init_extrema, xlabel="lx", ylabel="ly", title="linear diffusion (it_t=$it_t, it_τ=$it_τ)")
                        heatmap(xc[2:end-1], yc[2:end-1], H_v[:,:,nz_v÷2]; opts...)
                        frame(anim)
                    end
                end
            end

            it_τ += 1
        end

        # update physical time step
        ittot += it_τ; it_t += 1; t += dt
        Hᵗ .= H
        if isnan(err) error("NaN") end
    end

    # Calculate performance
    @static if BENCHMARK
        t_toc = Base.time() - t_tic                            # Stop the clock
        reads = length(Hᵗ)                                     # Read Only Memory Access: Hᵗ
        updates = length(H) + length(dHdt)                     # Update Memory access: H and dHdt
        A_eff = 1e-9 * (2 * updates + reads) * sizeof(Float64) # Effective main memory access per iteration [GB]
        t_it  = t_toc/(ittot-warmup)                           # Execution time per iteration [s]
        T_eff = A_eff/t_it                                     # Effective memory throughput [GB/s]
        println("time(sec)=$t_toc T_eff=$T_eff")
    end
    @printf("Ttime steps = %d, nx = %d, iterations tot = %d \n", it_t, nx, ittot)

    # Postprocessing 
    finalize_global_grid()

    # Create a gif movie on process 0
    @static if VISUALIZE
        if (me == 0) 
            gif(anim, "diffusion3D_multixpu.gif", fps = ((ittot ÷ nout) ÷ 5)) 
        end
    end

    # Return T_eff, t_toc nprocs and me if BENCHMARK and Result otherwise
    @static if BENCHMARK
        return T_eff, t_toc, nprocs, me
    else
        return xc, Array(H)
    end
end