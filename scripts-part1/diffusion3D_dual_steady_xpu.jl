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


@views function diffusion_3D()
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
    gr()
    ENV["GKSwstype"]="nul"
    anim = Animation();
    H_v = zeros(nx, ny, nz)

    t = 0.0; it = 0; ittot = 0

    damp  = 1-29/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Physical time loop
    while t<ttot
        iter = 0
        err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            
            @parallel diffusion3D_step!(H2, H, Hold, ResH, dHdt, dHdt2, D, damp, dt, dτ, dx, dy, dz)
            H, H2 = H2, H
            dHdt, dHdt2 = dHdt2, dHdt
            
            # Vizualise and calc error
            iter += 1
            if (iter % nout == 0)
                H_v .= H
                heatmap(xc, yc, H_v[:,:,nz÷2], aspect_ratio=1, label="H final", xlims=extrema(xc), ylims=extrema(yc), clims=extrema(H0), xlabel="lx", ylabel="H", title="linear diffusion (nt=$it, iters=$ittot)");frame(anim)
                err = norm(ResH)/sqrt(length(nx))
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end

    @printf("Ttime steps = %d, nx = %d, iterations tot = %d \n", it, nx, it)
    gif(anim, "diffusion3D_xpu.gif", fps = 15)
    return xc, H
end

xc, H = diffusion_3D()