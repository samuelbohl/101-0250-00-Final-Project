using Printf, LinearAlgebra, Plots

@views inn(A) = A[2:end-1,2:end-1,2:end-1]
@views d2_xi(A) = diff(diff(A[:,2:end-1,2:end-1], dims=1), dims=1)
@views d2_yi(A) = diff(diff(A[2:end-1,:,2:end-1], dims=2), dims=2)
@views d2_zi(A) = diff(diff(A[2:end-1,2:end-1,:], dims=3), dims=3)

@views function diffusion_3D(;steady=false)
    # Physics
    lx    = 10.0             # domain size
    ly    = 10.0             # domain size
    lz    = 10.0             # domain size
    D     = 1.0              # diffusion coefficient
    ttot  = 1.0              # total simulation time
    dt    = 0.2              # physical time step
    # Numerics
    nx, ny, nz    = 32, 32, 32
    tol   = 1e-8             # tolerance
    itMax = 1e5              # max number of iterations
    nout  = 10               # tol check
    # Derived numerics    
    dx,dy,dz    = lx/nx, ly/ny , lz/nz           # cell sizes
    dtau  = (1.0/(dx^2/D/6.1) + 1.0/dt)^-1 # iterative timestep

    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    qHx   = zeros(nx-1, ny-2, nz-2)
    qHy   = zeros(nx-2, ny-1, nz-2)
    qHz   = zeros(nx-2, ny-2, nz-1)
    dHdt  = zeros(nx-2, ny-2, nz-2)
    ResH  = zeros(nx-2, ny-2, nz-2)
    # Initial condition
    H0    = [2.0 * exp(-(xc[ix] - 0.5*lx)^2 -(yc[iy] - 0.5*ly)^2 -(zc[iz] - 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz]
    H     = copy(H0)
    Hold  = copy(H0)
    # Boundary Condition
    H[:,:,1]   .= 0
    H[:,:,end] .= 0
    H[:,1,:]   .= 0
    H[:,end,:] .= 0
    H[1,:,:]   .= 0
    H[end,:,:] .= 0

    t = 0.0; it = 0; ittot = 0
    if !steady
        println("Running with dual-time")
        damp  = 1-29/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)
        # Physical time loop
        while t<ttot
            iter = 0; err = 2*tol
            # Pseudo-transient iteration
            while err>tol && iter<itMax
                ResH       .= .-(inn(H) .- inn(Hold))./dt .+ D.*(d2_xi(H)./dx^2 .+ d2_yi(H)./dy^2 .+ d2_zi(H)./dz^2)
                dHdt       .= ResH   .+ damp .* dHdt
                H[2:end-1,2:end-1,2:end-1] .= inn(H) .+ dtau .* dHdt
                iter += 1; if (iter % nout == 0)  err = norm(ResH)/sqrt(length(nx))  end
            end
            ittot += iter; it += 1; t += dt
            Hold .= H
            if isnan(err) error("NaN") end
        end
    else
        println("Running in pseudo-time to steady-state")
        damp  = 1-4/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)
        # Pseudo-transient iteration
        iter = 0; err = 2*tol
        while err>tol && iter<itMax
            qHx        .= D.*diff(H[:, 2:end-1, 2:end-1], dims=1)./dx
            qHy        .= D.*diff(H[2:end-1, :, 2:end-1], dims=2)./dy
            qHz        .= D.*diff(H[2:end-1, 2:end-1, :], dims=3)./dz
            ResH       .= (diff(qHx, dims=1)./dx .+ diff(qHy, dims=2)./dy .+ diff(qHz, dims=3)./dz)
            dHdt       .= ResH   .+ damp .* dHdt
            H[2:end-1,2:end-1,2:end-1] .= inn(H) .+ dtau .* dHdt
            iter += 1; if (iter % nout == 0)  err = norm(ResH)/sqrt(length(nx))  end
        end
        ittot += iter; it += 1
        if isnan(err) error("NaN") end
    end
    @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=3), it, nx, ittot)
    # Visualise
    #p1 = heatmap(xc, yc, H0[:,:,nz÷2], linewidth=3, label="H initial")
    #p2 = heatmap(xc, yc, H[:,:,nz÷2], linewidth=3, label="H final", framestyle=:box, xlims=extrema(xc), ylims=extrema(H0), xlabel="lx", ylabel="H", title="linear diffusion (nt=$it, iters=$ittot)")
    #display(plot(p1, p2))
    return xc, H
end

xc, H = diffusion_3D()
