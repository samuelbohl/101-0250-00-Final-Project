using Printf, LinearAlgebra, Plots
using ImplicitGlobalGrid
import MPI

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(H2, H, ResH, dHdt, dHdt2, D, damp, dτ, dx, dy, dz)
    @all(ResH)  = (D*(@d2_xi(H)/dx^2 + @d2_yi(H)/dy^2 + @d2_zi(H)/dz^2))
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
    nout       = 20               # tol check
    # Derived numerics    
    dx, dy, dz = lx/nx, ly/ny, lz/nz             # cell sizes
    me, dims   = init_global_grid(nx, ny, nz);
    dx         = lx/(nx_g()-1);                              # Space step in x-dimension
    dy         = ly/(ny_g()-1);                              # Space step in y-dimension
    dz         = lz/(nz_g()-1);                              # Space step in z-dimension
    dτ         = (1.0/(dx^2/D/2.1) + 10.0/dt)^-1 # iterative timestep
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = copy(dHdt)
    ResH  = @zeros(nx-2, ny-2, nz-2)
    # Initial condition
    H     = @zeros(nx, ny, nz)
    H     = Data.Array([2.0 * exp(-(x_g(ix,dx,H)- 0.5*lx)^2 -(y_g(iy,dy,H) - 0.5*ly)^2 -(z_g(iz,dz,H)- 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
    H2    = copy(H)

      # Preparation of visualisation
      gr()
      ENV["GKSwstype"]="nul"
      anim = Animation();
      nx_v = (nx-2)*dims[1];
      ny_v = (ny-2)*dims[2];
      nz_v = (nz-2)*dims[3];
      H_v  = zeros(nx_v, ny_v, nz_v);
      H_nohalo = zeros(nx-2, ny-2, nz-2);

    println("Running in pseudo-time to steady-state")
    damp  = 1-4/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)

    # Pseudo-transient iteration
    t = 0.0; it = 0; err = 2*tol
    while err>tol && it<itMax

        if it % nout == 0
            H_nohalo .= H[2:end-1,2:end-1,2:end-1];                                           # Copy data to CPU removing the halo.
            gather!(H_nohalo, H_v)                                                            # Gather data on process 0 (could be interpolated/sampled first)
            if (me==0) heatmap(transpose(H_v[:,:,nz_v÷2]), aspect_ratio=1);  frame(anim); end  # Visualize it on process 0.
        end
        @hide_communication (16, 2, 2) begin
            @parallel diffusion3D_step!(H2, H, ResH, dHdt, dHdt2, D, damp, dτ, dx, dy, dz)
            update_halo!(H2);
        end
        H, H2 = H2, H
        dHdt, dHdt2 = dHdt2, dHdt
        it += 1

        # Vizualise and calc error
        if (it % nout == 0)  
            
            err = norm(ResH)/sqrt(length(nx))
        end

    end

    if isnan(err) error("NaN") end
    @printf("Ttime steps = %d, nx = %d, iterations tot = %d \n", it, nx, it)

    # Postprocessing
    if (me==0) gif(anim, "diffusion3D.gif", fps = 15) end                                     # Create a gif movie on process 0.
    finalize_global_grid();
    return xc, H
end

diffusion_3D()