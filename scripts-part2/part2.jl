using Plots

@views function acoustic_2D()
    # Physics
    Lx, Ly, Lz = 10.0, 10.0, 10.0
    ρ          = 1.0
    μ          = 10.0
    K          = 1.0
    ttot       = 20.0
    # Numerics
    nx, ny, nz = 64, 65, 66
    nout       = 1
    # Derived numerics
    dx, dy, dz = Lx/nx, Ly/ny, Lz,nz
    dt         = min(dx,dy,dz)/sqrt((K + 4/3*μ)/ρ)/2.1
    nt         = cld(ttot, dt)
    xc, yc, zc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny), LinRange(dz/2, Lz-dz/2, nz)
    # Array initialisation
    P      = exp.([-1.0*((x-Lx/2)^2 + (y-Ly/2)^2 + (z-Lz/2)^2) for x=xc, y=yc, z=zc])
    dPdt   = zeros(Float64, nx  , ny  , nz  )

    Vx     = zeros(Float64, nx+1, ny  , nz  )
    Vy     = zeros(Float64, nx  , ny+1, nz  )
    Vz     = zeros(Float64, nx  , ny  , nz+1)

    dVxdt  = zeros(Float64, nx-1, ny-2, nz-2)
    dVydt  = zeros(Float64, nx-2, ny-1, nz-2)
    dVzdt  = zeros(Float64, nx-2, ny-2, nz-1)

    τxx    = zeros(Float64, nx  , ny  , nz  )
    τyy    = zeros(Float64, nx  , ny  , nz  )
    τzz    = zeros(Float64, nx  , ny  , nz  )

    τxy    = zeros(Float64, nx-1, ny-1, nz-2)
    τyz    = zeros(Float64, nx-2, ny-1, nz-1)
    τzx    = zeros(Float64, nx-1, ny-2, nz-1)

    ∇V     = zeros(Float64, nx  , ny  , nz  )
    # Time loop
    for it = 1:nt
        τxx   .= τxx .+ dt*(2.0.*μ.*(diff(Vx,dims=1)/dx .- 1/3 .*∇V))
        τyy   .= τyy .+ dt*(2.0.*μ.*(diff(Vy,dims=2)/dy .- 1/3 .*∇V))
        τzz   .= τzz .+ dt*(2.0.*μ.*(diff(Vz,dims=3)/dz .- 1/3 .*∇V))

        τxy   .= τxy .+ dt*(μ.*(diff(Vx[2:end-1,:,2:end-1],dims=2)/dy .+ diff(Vy[:,2:end-1,2:end-1],dims=1)/dx))
        τyz   .= τyz .+ dt*(μ.*(diff(Vy[2:end-1,2:end-1,:],dims=3)/dz .+ diff(Vz[2:end-1,:,2:end-1],dims=2)/dy))
        τzx   .= τzx .+ dt*(μ.*(diff(Vz[:,2:end-1,2:end-1],dims=1)/dx .+ diff(Vx[2:end-1,2:end-1,:],dims=3)/dz))

        dVxdt .= .-1.0./ρ.*(diff(P[:,2:end-1,2:end-1],dims=1)./dx .- diff(τxx[:,2:end-1,2:end-1],dims=1)./dx .- diff(τxy,dims=2)./dy .- diff(τzx,dims=3)./dz)
        dVydt .= .-1.0./ρ.*(diff(P[2:end-1,:,2:end-1],dims=2)./dy .- diff(τyy[2:end-1,:,2:end-1],dims=2)./dy .- diff(τxy,dims=1)./dx .- diff(τyz,dims=3)./dz)
        dVzdt .= .-1.0./ρ.*(diff(P[2:end-1,2:end-1,:],dims=3)./dz .- diff(τzz[2:end-1,2:end-1,:],dims=3)./dz .- diff(τyz,dims=2)./dy .- diff(τzx,dims=1)./dx)

        Vx[2:end-1,2:end-1,2:end-1] .= Vx[2:end-1,2:end-1,2:end-1] .+ dt.*dVxdt
        Vy[2:end-1,2:end-1,2:end-1] .= Vy[2:end-1,2:end-1,2:end-1] .+ dt.*dVydt
        Vz[2:end-1,2:end-1,2:end-1] .= Vz[2:end-1,2:end-1,2:end-1] .+ dt.*dVzdt
        ∇V    .= diff(Vx,dims=1)./dx .+ diff(Vy,dims=2)./dy .+ diff(Vz,dims=3)./dz

        dPdt  .= .-K.*∇V
        P     .= P .+ dt.*dPdt

        if it % nout == 0
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.1, 0.4), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, P[:,:,cld(nz,2)]'; opts...))
        end
    end
    return
end

acoustic_2D()