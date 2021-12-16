const USE_GPU = false

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots

@parallel function compute_τ!(Vx, Vy, Vz, ∇V, τxx, τyy, τzz, τxy, τyz, τzx, μ, dt, dx, dy, dz)
    @all(τxx) = @all(τxx) + dt*(2.0*μ*(@d_xa(Vx)/dx) - 1.0/3.0*@all(∇V))
    @all(τyy) = @all(τyy) + dt*(2.0*μ*(@d_ya(Vy)/dy) - 1.0/3.0*@all(∇V))
    @all(τzz) = @all(τzz) + dt*(2.0*μ*(@d_za(Vz)/dz) - 1.0/3.0*@all(∇V))
    @all(τxy) = @all(τxy) + dt*μ*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(τyz) = @all(τyz) + dt*μ*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)
    @all(τzx) = @all(τzx) + dt*μ*(@d_xi(Vz)/dx + @d_zi(Vx)/dz)
    return
end

@parallel function compute_dV!(P, dVxdt, dVydt, dVzdt, τxx, τyy, τzz, τxy, τyz, τzx, ρ, dx, dy, dz)
    @all(dVxdt) = -1.0/ρ*(@d_xi(P)/dx - @d_xi(τxx)/dx - @d_ya(τxy)/dy - @d_za(τzx)/dz) 
    @all(dVydt) = -1.0/ρ*(@d_yi(P)/dy - @d_yi(τyy)/dy - @d_za(τyz)/dz - @d_xa(τxy)/dx)
    @all(dVzdt) = -1.0/ρ*(@d_zi(P)/dz - @d_zi(τzz)/dz - @d_xa(τzx)/dx - @d_ya(τyz)/dy)
    return
end

@parallel function compute_V!(Vx, Vy, Vz, ∇V, dVxdt, dVydt, dVzdt, dt, dx, dy, dz)
    @inn(Vx) = @inn(Vx) + dt*@all(dVxdt)
    @inn(Vy) = @inn(Vy) + dt*@all(dVydt)
    @inn(Vz) = @inn(Vz) + dt*@all(dVzdt)
    @all(∇V) = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
    return
end

@parallel function compute_P!(P, dPdt, ∇V, K, dt)
    @all(dPdt) = -K*@all(∇V)
    @all(P) = @all(P) + dt*@all(dPdt)
    return
end

@views function elastic_wave_3D()
    # Physics
    Lx, Ly, Lz = 10.0, 10.0, 10.0
    ρ          = 1.0
    μ          = 1.0
    K          = 1.0
    ttot       = 1.0
    # Numerics
    nx, ny, nz = 32, 32, 32
    nout       = 1
    # Derived numerics
    dx, dy, dz = Lx/nx, Ly/ny, Lz,nz
    dt         = min(dx,dy,dz)/sqrt((K + 4/3*μ)/ρ)/2.1
    nt         = cld(ttot, dt)
    xc, yc, zc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny), LinRange(dz/2, Lz-dz/2, nz)
    # Array initialisation
    P      = exp.([-1.0*((x-Lx/2)^2 + (y-Ly/2)^2 + (z-Lz/2)^2) for x=xc, y=yc, z=zc])
    dPdt   = @zeros(nx, ny, nz)

    Vx = @zeros(nx+1,ny  ,nz  )
    Vy = @zeros(nx  ,ny+1,nz  )
    Vz = @zeros(nx  ,ny  ,nz+1)
    ∇V = @zeros(nx  ,ny  ,nz  )
    
    dVxdt = @zeros(nx-1, ny-2, nz-2)
    dVydt = @zeros(nx-2, ny-1, nz-2)
    dVzdt = @zeros(nx-2, ny-2, nz-1)

    τxx = @zeros(nx  ,ny  ,nz  )
    τyy = @zeros(nx  ,ny  ,nz  )
    τzz = @zeros(nx  ,ny  ,nz  )
    τxy = @zeros(nx-1,ny-1,nz-2)
    τyz = @zeros(nx-2,ny-1,nz-1)
    τzx = @zeros(nx-1,ny-2,nz-1)
    # Time loop
    for it = 1:nt
        @parallel compute_τ!(Vx, Vy, Vz, ∇V, τxx, τyy, τzz, τxy, τyz, τzx, μ, dt, dx, dy, dz)
        @parallel compute_dV!(P, dVxdt, dVydt, dVzdt, τxx, τyy, τzz, τxy, τyz, τzx, ρ, dx, dy, dz)
        @parallel compute_V!(Vx, Vy, Vz, ∇V, dVxdt, dVydt, dVzdt, dt, dx, dy, dz)
        @parallel compute_P!(P, dPdt, ∇V, K, dt)
        if it % nout == 0
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.15, 0.65), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, P[:,:,cld(nz,2)]'; opts...))
        end
    end
    return xc, P
end

xc, P = elastic_wave_3D()