using Printf, LinearAlgebra, Plots
using ImplicitGlobalGrid
import MPI

const USE_GPU = false
# IF GPU: ~/.julia/bin/mpiexecjl -n 1 julia --project ./scripts-part1/diffusion3D_dual_steady_multixpu.jl 
# IF CPU: ~/.julia/bin/mpiexecjl -n 12 julia --project ./scripts-part1/diffusion3D_dual_steady_multixpu.jl 
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
    nx, ny, nz = 34, 34, 34
    tol        = 1e-8             # tolerance
    itMax      = 1e5              # max number of iterations
    nout       = 10               # tol check
    # Derived numerics    
    dx, dy, dz = lx/nx, ly/ny, lz/nz             # cell sizes
    me, dims   = init_global_grid(nx, ny, nz);
    dx         = lx/(nx_g()-2);                              # Space step in x-dimension
    dy         = ly/(ny_g()-2);                              # Space step in y-dimension
    dz         = lz/(nz_g()-2);                              # Space step in z-dimension
    dτ         = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1 # iterative timestep
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx-2), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = copy(dHdt)
    ResH  = @zeros(nx-2, ny-2, nz-2)
    # Initial condition
    H     = @zeros(nx, ny, nz)
    H     = Data.Array([2.0 * exp(-(x_g(ix,dx,H)- 0.5*lx)^2 -(y_g(iy,dy,H) - 0.5*ly)^2 -(z_g(iz,dz,H)- 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
    H2    = copy(H)
    Hold  = copy(H)

    # Preparation of visualisation
    gr()
    ENV["GKSwstype"]="nul"
    anim = Animation();
    nx_v = (nx-2)*dims[1];
    ny_v = (ny-2)*dims[2];
    nz_v = (nz-2)*dims[3];
    H_v  = zeros(nx_v, ny_v, nz_v);
    H_nohalo = zeros(nx-2, ny-2, nz-2);

    t = 0.0; it = 0; ittot = 0

    damp  = 1-29/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Physical time loop
    while t<ttot
        iter = 0
        err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax
            
            @parallel diffusion3D_step!(H2, H, Hold, ResH, dHdt, dHdt2, D, damp, dt, dτ, dx, dy, dz)
            update_halo!(H2);
            H, H2 = H2, H
            dHdt, dHdt2 = dHdt2, dHdt
            
            # Vizualise and calc error
            iter += 1
            if iter % nout == 0
                err = norm(ResH)/sqrt(length(nx))
                H_nohalo .= H[2:end-1,2:end-1,2:end-1];                                           # Copy data to CPU removing the halo.
                gather!(H_nohalo, H_v)                                                            # Gather data on process 0 (could be interpolated/sampled first)
                if (me==0) heatmap(H_v[:,:,nz÷2], aspect_ratio=1, label="H final", xlims=extrema(xc), ylims=extrema(yc), clims=extrema(H_v), xlabel="lx", ylabel="H", title="linear diffusion (nt=$it, iters=$ittot)");frame(anim); end  # Visualize it on process 0.
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("NaN") end
    end

    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=3), it, nx, ittot)

    # Postprocessing
    if (me==0) gif(anim, "diffusion3D_multixpu.gif", fps = 15) end                                     # Create a gif movie on process 0.
    finalize_global_grid();
    return xc, H_v
end

xc, H = diffusion_3D()