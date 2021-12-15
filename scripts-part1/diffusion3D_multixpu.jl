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

@views inn(A) = A[2:end-1,2:end-1,2:end-1]

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

norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))


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
    me, dims   = init_global_grid(nx, ny, nz)
    dx         = lx/(nx_g());                              # Space step in x-dimension
    dy         = ly/(ny_g());                              # Space step in y-dimension
    dz         = lz/(nz_g());                              # Space step in z-dimension
    dτ         = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1            # iterative timestep

    # Array allocation
    dHdt  = @zeros(nx-2, ny-2, nz-2)
    dHdt2 = copy(dHdt)
    ResH  = @zeros(nx-2, ny-2, nz-2)
    # Initial condition
    H     = @zeros(nx, ny, nz)
    H0    = Data.Array([2.0 * exp(-(dx/2 + x_g(ix,dx,H)- 0.5*lx)^2 -(dy/2 + y_g(iy,dy,H) - 0.5*ly)^2 -(dz/2 + z_g(iz,dz,H)- 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
    H     = copy(H0)
    H2    = copy(H)
    Hold  = copy(H)

    # Preparation of visualisation
    gr()
    ENV["GKSwstype"]="nul"
    anim = Animation()
    xc   = LinRange(dx/2, lx-dx/2, nx_g()) 
    yc   = LinRange(dy/2, ly-dy/2, ny_g())
    nx_v = (nx-2)*dims[1]
    ny_v = (ny-2)*dims[2]
    nz_v = (nz-2)*dims[3]
    H_v  = zeros(nx_v, ny_v, nz_v)
    H_nohalo = zeros(nx-2, ny-2, nz-2)
    

    t = 0.0; it = 0; ittot = 0

    damp  = 1-29/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Physical time loop
    while t<ttot
        iter = 0
        err = 2*tol
        # Pseudo-transient iteration
        while err>tol && iter<itMax

            @parallel compute_ResH!(H, Hold, ResH, D, dt, dx, dy, dz)
            @parallel compute_dHdt!(ResH, dHdt, dHdt2, damp)
            dHdt, dHdt2 = dHdt2, dHdt
            @parallel compute_H!(H2, H, dHdt, dτ)
            update_halo!(H2)
            H, H2 = H2, H
            
            # Vizualise and calc error
            iter += 1
            if iter % nout == 0
                err = norm_g(ResH)/sqrt(length(ResH))

                H_nohalo .= H[2:end-1,2:end-1,2:end-1]                                            # Copy data to CPU removing the halo.
                gather!(H_nohalo, H_v)                                                            # Gather data on process 0 (could be interpolated/sampled first)
                opts = (aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), clims=extrema(H0), xlabel="lx", ylabel="ly", title="linear diffusion (nt=$it, iters=$iter)")                                        
                if (me==0) heatmap(xc[2:end-1], yc[2:end-1], H_v[:,:,nz_v÷2]; opts...); frame(anim) end               # Visualize it on process 0.
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


    return xc, Array(H)
end