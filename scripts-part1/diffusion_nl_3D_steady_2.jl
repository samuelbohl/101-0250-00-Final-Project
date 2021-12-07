using Plots, Statistics, LinearAlgebra

@views function diffusion_3D()
    # Physics
    Lx, Ly, Lz = 10.0, 10.0, 10.0
    ttot   = 1.0

    # Numerics
    nx, ny, nz = 32, 32, 32
    nout = 1
    epsi = 1e-18

    # Derived numerics
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    xc, yc, zc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny), LinRange(dz/2, Lz-dz/2, nz)
    dmp      = 1.0 - 2π/nx

    # Array initialisation
    H    = zeros(nx, ny, nz)
    H    = [2 .* exp(-(xc[ix] - Lx/2)^2 -(yc[iy] - Ly/2)^2 -(zc[iz] - Lz/2)^2) for ix=1:nx, iy=1:ny, iz=1:nz]
    display(plot([H[it, it, it] for it=1:nx]))
    dHdt = zeros(nx-2, ny-2, nz-2)
    qx   = zeros(nx-1, ny-2, nz-2)
    qy   = zeros(nx-2, ny-1, nz-2)
    qz   = zeros(nx-2, ny-2, nz-1)
    opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:davos, xlabel="Lx", ylabel="Ly")
    p1 = heatmap(transpose(H[:,:,16]); opts...)
    display(p1)

    # Time loop
    it = 0; t = 0.0; err = 1.0
    while err > epsi && t < ttot
        dt          = 0.2
        qx         .= .-diff(H[:, 2:end-1, 2:end-1], dims=1)./dx
        qy         .= .-diff(H[2:end-1, :, 2:end-1], dims=2)./dy
        qz         .= .-diff(H[2:end-1, 2:end-1, :], dims=3)./dz

        dHdt       .= .-(diff(qx, dims=1)./dx .+ diff(qy, dims=2)./dy .+ diff(qz, dims=3)./dz) .+ dHdt .* dmp
        
        H[2:end-1, 2:end-1, 2:end-1] .= H[2:end-1, 2:end-1, 2:end-1] .+ dt .* dHdt
        
        it += 1; t += dt
        
        # calculate error
        ∆H = dt.*dHdt
        err = norm(∆H)/sqrt(length(∆H))
        
        # Vizualization
        if it % nout == 0
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(t, sigdigits=3))s, it = $(it) err = $(round(err, sigdigits=3))")
            p1 = heatmap(transpose(H[:,:,16]); opts...)
            p1 = display(plot([H[16, it, 16] for it=1:nx]))
            display(p1)
        end

    end

    return xc, H
end

xc_g, H = diffusion_3D()