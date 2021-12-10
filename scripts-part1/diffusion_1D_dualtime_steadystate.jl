using Printf, LinearAlgebra, Plots

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
    dτ         = (1.0/(dx^2/D/2.1) + 10.0/dt)^-1 # iterative timestep
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    qHx   = zeros(nx-1, ny-2, nz-2)
    qHy   = zeros(nx-2, ny-1, nz-2)
    qHz   = zeros(nx-2, ny-2, nz-1)
    dHdt  = zeros(nx-2, ny-2, nz-2)
    ResH  = zeros(nx-2, ny-2, nz-2)
    # Initial condition
    H0    = [2.0 * exp(-(xc[ix] - 0.5*lx)^2 -(yc[iy] - 0.5*ly)^2 -(zc[iz] - 0.5*lz)^2) for ix=1:nx, iy=1:ny, iz=1:nz]
    H     = ones(nx, ny, nz) .* H0
     ittot = 0

    println("Running in pseudo-time to steady-state")
    damp  = 1-4/nx          # damping (this is a tuning parameter, dependent on e.g. grid resolution)

    # Pseudo-transient iteration
    t = 0.0; it = 0; err = 2*tol
    while err>tol && it<itMax
        qHx        .= .- D.*diff(H[:, 2:end-1, 2:end-1], dims=1)./dx
        qHy        .= .- D.*diff(H[2:end-1, :, 2:end-1], dims=2)./dy
        qHz        .= .- D.*diff(H[2:end-1, 2:end-1, :], dims=3)./dz
        ResH       .= .-(diff(qHx, dims=1)./dx .+ diff(qHy, dims=2)./dy .+ diff(qHz, dims=3)./dz)
 
        dHdt       .= ResH .+ damp .* dHdt
        H[2:end-1,2:end-1,2:end-1]  .= H[2:end-1, 2:end-1, 2:end-1] .+ dτ .* dHdt
        
        it += 1

        # Vizualise and calc error
        if (it % nout == 0)  
            display(heatmap(xc, yc, H[:,:,nz÷2], linewidth=3, label="H final", framestyle=:box, xlabel="lx", ylabel="ly", title="3D diffusion (iters=$it)"))
            err = norm(ResH)/sqrt(length(nx))
        end

    end

    if isnan(err) error("NaN") end
    @printf("Ttime steps = %d, nx = %d, iterations tot = %d \n", it, nx, it)
    return xc, H
end

diffusion_3D()