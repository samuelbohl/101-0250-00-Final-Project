using Printf, LinearAlgebra, Plots, ParallelStencil, ParallelStencil.FiniteDifferences3D

# Initialize Globals
if !@isdefined USE_GPU
    const USE_GPU = false
end

if !@isdefined BENCHMARK
    const BENCHMARK = true
end

if !@isdefined VISUALIZE
    const VISUALIZE = false
end

if !@isdefined STEADY
    const STEADY = false
end

# Initialize `ParallelStencil.jl`
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

# Macro for shorter notation
macro inner(A) esc(:( $A[2:end-1,2:end-1,2:end-1] )) end

@parallel function compute_ResH!(H, Hold, ResH, _dt, D_dx², D_dy², D_dz²)
    @all(ResH) = -(@inn(H) - @inn(Hold)) * _dt + (@d2_xi(H)*D_dx² + @d2_yi(H)*D_dy² + @d2_zi(H)*D_dz²)
    return
end

@parallel function compute_ResH_steady!(H, ResH, D_dx², D_dy², D_dz²)
    @all(ResH) = @d2_xi(H)*D_dx² + @d2_yi(H)*D_dy² + @d2_zi(H)*D_dz²
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
    diffusion_3D(res[,tol])

Runs the 3D dualtime diffusion simulation

# Arguments
- `res::Int`: xyz-resolutions of the simulation
- `tol::Float64`(optional): tolerance for convergence - default: `1e-8`

# Returns (Depending on globals)
- If BENCHMARK is true -> 
    `T_eff`: Effective memory throughput [GB/s]
- If STEADY is true and BENCHMARK is false -> 
    `ittot`, `xc` and `H`: Total number of iterations needed to reach steadystate, the global x-coordinate Array and the global solution array
- If STEADY and BENCHMARK are false -> 
    `xc` and `H`: The global x-coordinate Array and the global solution array
"""
@views function diffusion_3D(res::Int; tol=1e-8)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D     = 1.0                   # diffusion coefficient
    ttot  = 1.0                   # total simulation time
    dt    = 0.2                   # physical time step

    # Numerics
    nx, ny, nz = res, res, res
    itMax      = 1e4              # max number of iterations
    nout       = 10               # tol check

    # Derived numerics    
    dx, dy, dz = lx/nx, ly/ny, lz/nz            # cell sizes
    dτ         = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1 # iterative timestep
    damp       = 1-29/nx                        # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    xc         = LinRange(dx/2, lx-dx/2, nx)    #  
    yc         = LinRange(dy/2, ly-dy/2, ny)
    zc         = LinRange(dz/2, lz-dz/2, nz)
    _dt        = 1.0/dt
    D_dx²      = D/dx^2
    D_dy²      = D/dy^2
    D_dz²      = D/dz^2

    # Array allocation
    H     = @zeros(nx  , ny  , nz  )
    H2    = @zeros(nx  , ny  , nz  )
    Hᵗ    = @zeros(nx  , ny  , nz  )
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = @zeros(nx-2, ny-2, nz-2)
    ResH  = @zeros(nx-2, ny-2, nz-2)

    # Initial condition
    @inner(H) .= @inner(Data.Array([2.0 * exp(-(xc[ix] - 0.5*lx)^2 -(yc[iy] - 0.5*ly)^2 -(zc[iz] - 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz]))
    H2        .= H
    Hᵗ        .= H

    # Benchmark variables
    @static if BENCHMARK
        warmup = 10         # Number of warmup-iterations
        t_tic  = 0.0        # Holds start time
        itMax  = 1e4/nx     # scale max number of iterations down my grid size
    end

    # Run to Steady State variables
    @static if STEADY
        damp = 1-4/nx
    end

    # Animation object for visualization
    @static if VISUALIZE
        gr()
        ENV["GKSwstype"] = "nul"
        anim = Animation()
        H_v = zeros(nx, ny, nz)
        init_extrema = extrema(H)
    end

    # initial physical time loop conditions
    t_tic = 0.0
    t = 0.0
    it_t = 0
    ittot = 0

    @static if !STEADY
        println("Running with dual-time")

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
                H, H2 = H2, H # Pointer Swap

                # Calculate L2 norm
                if (it_τ % nout == 0)
                    err = norm(ResH)/sqrt(length(ResH))
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
                        H_v .= H
                        opts = (aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), clims=init_extrema, xlabel="lx", ylabel="ly", title="3D Diffusion (it_t=$it_t, it_τ=$it_τ)")                                        
                        heatmap(xc, yc, H_v[:,:,nz÷2]; opts...)
                        frame(anim)
                    end
                end

                it_τ += 1
            end

            # update physical time step
            ittot += it_τ; it_t += 1; t += dt
            Hᵗ .= H
            if isnan(err) error("NaN") end
        end
    else 
        println("Running in pseudo-time to steady-state")

        # initial pseudo-transient loop conditions
        it_τ = 0
        err = 2*tol

        # Pseudo-transient iteration
        while err>tol && it_τ<itMax
            # Step 1
            @parallel compute_ResH_steady!(H, ResH, D_dx², D_dy², D_dz²)
            # Step 2
            @parallel compute_dHdt!(ResH, dHdt, dHdt2, damp)
            dHdt, dHdt2 = dHdt2, dHdt # Pointer Swap
            # Step 3
            @parallel compute_H!(H2, H, dHdt, dτ)
            H, H2 = H2, H # Pointer Swap

            # Calculate L2 norm (error)
            if (it_τ % nout == 0)
                err = norm(ResH)/sqrt(length(ResH))
            end
            it_τ += 1
        end

        # update physical time step
        ittot += it_τ; it_t += 1
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

    # Save the animation as a GIF
    @static if VISUALIZE
        gif(anim, "diffusion3D_xpu.gif", fps = 15)
    end

    # Return T_eff if BENCHMARK and Result otherwise
    @static if BENCHMARK
        return T_eff
    elseif STEADY
        return ittot, xc, Array(H)
    else
        return xc, Array(H)
    end
end