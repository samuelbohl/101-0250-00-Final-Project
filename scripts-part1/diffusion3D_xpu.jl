using Printf, LinearAlgebra, Plots

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(H2, H, Hold, ResH, dHdt, dHdt2, D, damp, dt, dτ, dx, dy, dz)
    @all(ResH)  = -(@inn(H) - @inn(Hold))/dt + (D*(@d2_xi(H)/dx^2 + @d2_yi(H)/dy^2 + @d2_zi(H)/dz^2))
    @all(dHdt2) = @all(ResH) + damp * @all(dHdt)
    @inn(H2)    = @inn(H) + dτ * @all(dHdt2)
    return
end

@parallel function compute_ResH!(H, Hold, ResH, D, dt, dx, dy, dz)
    @all(ResH)  = -(@inn(H) - @inn(Hold))/dt + (D*(@d2_xi(H)/dx^2 + @d2_yi(H)/dy^2 + @d2_zi(H)/dz^2))
    return
end

@parallel function compute_dHdt!(ResH, dHdt, dHdt2, damp)
    @all(dHdt2) = @all(ResH) + damp * @all(dHdt)
    return
end
@parallel function compute_H!(H2, H, dHdt, dτ)
    @inn(H2)    = @inn(H) + dτ * @all(dHdt)
    return
end


@views function diffusion_3D(;do_visu=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0 # domain size
    D     = 1.0                   # diffusion coefficient
    ttot  = 1.0                   # total simulation time
    dt    = 0.2                   # physical time step
    # Numerics
    nx, ny, nz = 32, 32, 32
    tol        = 1e-8             # tolerance
    itMax      = 1e5              # max number of iterations
    nout       = 10               # tol check
    # Derived numerics    
    dx, dy, dz = lx/nx, ly/ny, lz/nz             # cell sizes
    dτ         = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1 # iterative timestep
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = copy(dHdt)
    ResH  = @zeros(nx-2, ny-2, nz-2)
    R2 = copy(ResH)
    # Initial condition
    H0    = Data.Array([2.0 * exp(-(xc[ix] - 0.5*lx)^2 -(yc[iy] - 0.5*ly)^2 -(zc[iz] - 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
    H     = copy(H0)
    H2    = copy(H0)
    Hold  = copy(H0)

    # Preparation of visualisation
    if (do_visu)
        gr()
        ENV["GKSwstype"]="nul"
        anim = Animation();
        H_v = zeros(nx, ny, nz)
    end

    t_tic = 0.0;t = 0.0; it = 0; ittot = 0

    damp  = 1-29/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Physical time loop
    while t<ttot
        iter = 0
        err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax

            if (iter == 11 && ittot == 0) t_tic = Base.time(); end
            #@parallel diffusion3D_step!(H2, H, Hold, ResH, dHdt, dHdt2, D, damp, dt, dτ, dx, dy, dz)
            @parallel compute_ResH!(H, Hold, ResH, D, dt, dx, dy, dz)
            @parallel compute_dHdt!(ResH, dHdt, dHdt2, damp)
            dHdt, dHdt2 = dHdt2, dHdt
            @parallel compute_H!(H2, H, dHdt, dτ)
            H, H2 = H2, H
            
            # Vizualise and calc error
            iter += 1
            if (iter % nout == 0)
                if do_visu
                    H_v .= H
                    opts = (aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), clims=extrema(H0), xlabel="lx", ylabel="ly", title="3D Diffusion (nt=$it, iters=$iter)")                                        
                    heatmap(xc, yc, H_v[:,:,nz÷2]; opts...); frame(anim)
                end
                err = norm(ResH)/sqrt(length(ResH))
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end

    t_toc = Base.time() - t_tic
    # Performance
    A_eff = 9/1e9*nx*ny*nz*sizeof(Float64);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    t_it  = t_toc/(ittot-10);                    # Execution time per iteration [s]
    T_eff = A_eff/t_it;                          # Effective memory throughput [GB/s]
    println("time(sec)=$t_toc T_eff=$T_eff")

    @printf("Ttime steps = %d, nx = %d, iterations tot = %d \n", it, nx, it)
    if (do_visu) gif(anim, "diffusion3D_xpu.gif", fps = 15) end
    return xc, H
end