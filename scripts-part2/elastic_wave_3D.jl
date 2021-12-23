using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots

@parallel function compute_τ!(Vx, Vy, Vz, ∇V, τxx, τyy, τzz, τxy, τyz, τzx, dt1_3, dtμ_dx, dtμ_dy, dtμ_dz, dt2μ_dx, dt2μ_dy, dt2μ_dz)
    @all(τxx) = @all(τxx) + @d_xi(Vx)*dt2μ_dx - dt1_3*@inn_yz(∇V)
    @all(τyy) = @all(τyy) + @d_yi(Vy)*dt2μ_dy - dt1_3*@inn_xz(∇V)
    @all(τzz) = @all(τzz) + @d_zi(Vz)*dt2μ_dz - dt1_3*@inn_xy(∇V)
    @all(τxy) = @all(τxy) + @d_yi(Vx)*dtμ_dy + @d_xi(Vy)*dtμ_dx
    @all(τyz) = @all(τyz) + @d_zi(Vy)*dtμ_dz + @d_yi(Vz)*dtμ_dy
    @all(τzx) = @all(τzx) + @d_xi(Vz)*dtμ_dx + @d_zi(Vx)*dtμ_dz
    return
end

@parallel function compute_V!(P, Vx, Vy, Vz, ∇V, τxx, τyy, τzz, τxy, τyz, τzx, dt_ρ_dx, dt_ρ_dy, dt_ρ_dz)
	# dVxdt = -1/ρ*(@d_xi(P)*_dx - @d_xa(τxx)*_dx - @d_ya(τxy)*_dy - @d_za(τzx)*_dz)
	# Vx    = Vx + dt*dVxdt
    @inn(Vx) = @inn(Vx) - (@d_xi(P) - @d_xa(τxx))*dt_ρ_dx + @d_ya(τxy)*dt_ρ_dy + @d_za(τzx)*dt_ρ_dz
    @inn(Vy) = @inn(Vy) - (@d_yi(P) - @d_ya(τyy))*dt_ρ_dy + @d_za(τyz)*dt_ρ_dz + @d_xa(τxy)*dt_ρ_dx
    @inn(Vz) = @inn(Vz) - (@d_zi(P) - @d_za(τzz))*dt_ρ_dz + @d_xa(τzx)*dt_ρ_dx + @d_ya(τyz)*dt_ρ_dy
    return
end

@parallel function compute_∇V!(Vx, Vy, Vz, ∇V, _dx, _dy, _dz)
    @all(∇V) = @d_xa(Vx)*_dx + @d_ya(Vy)*_dy + @d_za(Vz)*_dz
    return
end

@parallel function compute_P!(P, ∇V, K, dt)
	# dPdt = -K*∇V
	# P    = P + dt*dPdt
    @all(P) = @all(P) - dt*K*@all(∇V)
    return
end

"""
Runs the 3D elastic wave simulation\\
`elastic_wave_3D(res, ttot) -> xc, P`\\
    `res`    (Int)     : xyz-resolutions of the simulation\\
    `ttot`   (Float64) : total simulation time\\
    `xc` and `P` are returned for testing purposes
"""
@views function elastic_wave_3D(res::Int, ttot::Float64)
    # Physics
    Lx, Ly, Lz = 10.0, 10.0, 10.0
    ρ          = 1.0
    μ          = 1.0
    K          = 1.0

    # Numerics
    nx, ny, nz = res, res, res

    # Derived numerics
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    dt         = min(dx,dy,dz)/sqrt((K + 4/3*μ)/ρ)/2.1
    nt         = cld(ttot, dt)
    xc, yc, zc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny), LinRange(dz/2, Lz-dz/2, nz)

	# Optimized numerics
	_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz
    dtμ_dx, dtμ_dy, dtμ_dz = dt*μ*_dx, dt*μ*_dy, dt*μ*_dz
    dt2μ_dx, dt2μ_dy, dt2μ_dz = 2.0*dtμ_dx, 2.0*dtμ_dy, 2.0*dtμ_dz
	dt_ρ = dt/ρ
    dt_ρ_dx, dt_ρ_dy, dt_ρ_dz = dt_ρ*_dx, dt_ρ*_dy, dt_ρ*_dz
    dt1_3 = dt/3.0

    # Array initialisation
    P   = Data.Array(exp.([-1.0*((x-Lx/2)^2 + (y-Ly/2)^2 + (z-Lz/2)^2) for x=xc, y=yc, z=zc]))
    Vx  = @zeros(nx+1,ny  ,nz  )
    Vy  = @zeros(nx  ,ny+1,nz  )
    Vz  = @zeros(nx  ,ny  ,nz+1)
    ∇V  = @zeros(nx  ,ny  ,nz  )
    τxx = @zeros(nx  ,ny-2,nz-2)
    τyy = @zeros(nx-2,ny  ,nz-2)
    τzz = @zeros(nx-2,ny-2,nz  )
    τxy = @zeros(nx-1,ny-1,nz-2)
    τyz = @zeros(nx-2,ny-1,nz-1)
    τzx = @zeros(nx-1,ny-2,nz-1)

    # Benchmark variables
    @static if BENCHMARK
        warmup = 10     # Number of warmup-iterations
        t_tic  = 0.0    # Holds start time
    end

    # Animation object for visualization
    @static if VISUALIZE
        anim = Animation()
    end

    # Time loop
    for it = 1:nt
        @parallel compute_τ!(Vx, Vy, Vz, ∇V, τxx, τyy, τzz, τxy, τyz, τzx, dt1_3, dtμ_dx, dtμ_dy, dtμ_dz, dt2μ_dx, dt2μ_dy, dt2μ_dz)
        @parallel compute_V!(P, Vx, Vy, Vz, ∇V, τxx, τyy, τzz, τxy, τyz, τzx, dt_ρ_dx, dt_ρ_dy, dt_ρ_dz)
        @parallel compute_∇V!(Vx, Vy, Vz, ∇V, _dx, _dy, _dz)
        @parallel compute_P!(P, ∇V, K, dt)

        # Start the clock after warmup-iterations
        @static if BENCHMARK
            if (it == warmup+1)
                t_tic = Base.time();
            end
        end

        # Render a slice of the 3D pressure-map and save as an animation frame
        @static if VISUALIZE
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.2, 0.6), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            heatmap(xc, yc, P[:,:,cld(nz,2)]'; opts...)
            frame(anim)
        end
    end

    # Calculate performance
    @static if BENCHMARK
        t_toc = Base.time() - t_tic                 # Stop the clock
        A_eff = 1e-9 * sizeof(Float64) * (          # Effective main memory access per iteration [GB]
              2 * (length(τxx) + length(τyy) + length(τzz) + length(τxy) + length(τyz) + length(τzx))
            + 2 * (length(Vx)  + length(Vy)  + length(Vz))
            + 2 * (length(P))
        )
        t_it  = t_toc/(nt-warmup)                   # Execution time per iteration [s]
        T_eff = A_eff/t_it                          # Effective memory throughput [GB/s]
        println("size=$res, t=$t_toc s, T_eff=$T_eff GB/s")
    end

    # Save the animation as a GIF
    @static if VISUALIZE
		fps = min(60.0, (float(nt)/ttot))
        gif(anim, "./docs/img/elastic_wave_3D.gif", fps=fps)
    end

	@static if BENCHMARK
		return T_eff
	else
		return xc, P
	end
end