# Part 1 of final project: Diffusion equation

@views function init()
    nx_g = 30             # number of global grid points
    ny_g = 30             # number of global grid points
    nz_g = 30             # number of global grid points
    xc_g = zeros(nx_g, 1) # global coord vector
    H    = zeros(nx_g, ny_g, nz_g) # global solution as obtained by implicitGlobalGrid's `gather!()`
    inds = Int.(ceil.(LinRange(1, length(xc_g), 12)))
    xc_g[inds]        .= [0.15625, 1.09375, 2.03125, 2.96875, 3.90625, 4.843750000000001, 5.468749999999999, 6.40625, 7.34375, 8.28125, 9.21875, 9.84375]
    H[inds, inds, 15] .= [6.711882385110737e-21 2.4514117276084958e-17 1.5437604567532982e-14 1.6762365118986786e-12 3.138212957188894e-11 1.0130275516790658e-10 8.332937341028532e-11 1.4367785881450301e-11 4.2714301545963265e-13 2.1895179117406884e-15 1.9351535299019154e-18 6.711882385110737e-21; 2.4514117276084958e-17 0.0004814874277834956 0.0014445717212427402 0.0032493045299438012 0.005557594231157323 0.006946731379668997 0.006690697700021068 0.004803038520579223 0.0025518750281671872 0.0010411002857436305 0.00029141318932587976 2.4514117276084958e-17; 1.5437604567532982e-14 0.0014445717212427402 0.004624363091812729 0.011159479624478753 0.020196376510907457 0.025913346399889084 0.02484567081606447 0.017173580755691007 0.008566902048912462 0.003255743161934395 0.0008608124023284637 1.5437604567532982e-14; 1.6762365118986786e-12 0.0032493045299438012 0.011159479624478753 0.029095533074715414 0.056125785858625855 0.07421776052483857 0.0707876802440615 0.04685098387699614 0.02177896579933202 0.007659758643077384 0.0019037581454307546 1.6762365118986786e-12; 3.138212957188894e-11 0.005557594231157323 0.020196376510907457 0.056125785858625855 0.11433141121531959 0.15522789812463358 0.1473718066389615 0.0939249428097469 0.04113536620718002 0.013581240422684215 0.0032124500970325837 3.138212957188894e-11; 1.0130275516790658e-10 0.006946731379668997 0.025913346399889084 0.07421776052483857 0.15522789812463358 0.21350368660148342 0.20223719296375614 0.12653170317321735 0.053851078219800314 0.017260273812497083 0.003990509147237947 1.0130275516790658e-10; 8.332937341028532e-11 0.006690697700021068 0.02484567081606447 0.0707876802440615 0.1473718066389615 0.20223719296375614 0.19164191533936806 0.12029222684071515 0.051452364828806635 0.01657647018690786 0.0038475611129833603 8.332937341028532e-11; 1.4367785881450301e-11 0.004803038520579223 0.017173580755691007 0.04685098387699614 0.0939249428097469 0.12653170317321735 0.12029222684071515 0.07752458731237559 0.03454914825401289 0.011616853176862237 0.0027870615866269714 1.4367785881450301e-11; 4.2714301545963265e-13 0.0025518750281671872 0.008566902048912462 0.02177896579933202 0.04113536620718002 0.053851078219800314 0.051452364828806635 0.03454914825401289 0.016437844426928104 0.005928856172534874 0.001503341539029276 4.2714301545963265e-13; 2.1895179117406884e-15 0.0010411002857436305 0.003255743161934395 0.007659758643077384 0.013581240422684215 0.017260273812497083 0.01657647018690786 0.011616853176862237 0.005928856172534874 0.0023115783364469583 0.0006238849765532687 2.1895179117406884e-15; 1.9351535299019154e-18 0.00029141318932587976 0.0008608124023284637 0.0019037581454307546 0.0032124500970325837 0.003990509147237947 0.0038475611129833603 0.0027870615866269714 0.001503341539029276 0.0006238849765532687 0.00017702766887648104 1.9351535299019154e-18; 6.711882385110737e-21 2.4514117276084958e-17 1.5437604567532982e-14 1.6762365118986786e-12 3.138212957188894e-11 1.0130275516790658e-10 8.332937341028532e-11 1.4367785881450301e-11 4.2714301545963265e-13 2.1895179117406884e-15 1.9351535299019154e-18 6.711882385110737e-21]
    return xc_g, H
end

xc_g, H = init();

# macros to avoid array allocation
macro qx(ix,iy)  esc(:( -D_dx*(C[$ix+1,$iy+1] - C[$ix,$iy+1]) )) end
macro qy(ix,iy)  esc(:( -D_dy*(C[$ix+1,$iy+1] - C[$ix+1,$iy]) )) end

@parallel_indices (ix,iy) function compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
    if (ix<=size_C1_2 && iy<=size_C2_2)
        C2[ix+1,iy+1] = C[ix+1,iy+1] - dt*( (@qx(ix+1,iy) - @qx(ix,iy))*_dx + (@qy(ix,iy+1) - @qy(ix,iy))*_dy )
    end
    return
end

@views function diffusion_2D(;do_visu=false, local_grid=32)
    # Physics
    Lx, Ly  = 10.0, 10.0
    D       = 1.0
    ttot    = 1e1
    # Numerics
    nx, ny, nz = local_grid,local_grid, local_grid # number of grid points
    nout    = 10
    # Derived numerics
    me, dims = init_global_grid(nx, ny, 1)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end  # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy  = Lx/nx_g(), Ly/ny_g()
    dt      = min(dx, dy)^2/D/4.1
    nt      = cld(ttot, dt)
    xc, yc  = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    D_dx    = D/dx
    D_dy    = D/dy
    _dx, _dy= 1.0/dx, 1.0/dy
    # Array initialisation
    C       = @zeros(nx,ny)
    C      .= Data.Array([exp(-(x_g(ix,dx,C)+dx/2 -Lx/2)^2 -(y_g(iy,dy,C)+dy/2 -Ly/2)^2) for ix=1:size(C,1), iy=1:size(C,2)])
    C2      = copy(C)
    size_C1_2, size_C2_2 = size(C,1)-2, size(C,2)-2
    # Preparation of visualisation
    if do_visu
        if (me==0) ENV["GKSwstype"]="nul"; if isdir("viz2D_mxpu_out")==false mkdir("viz2D_mxpu_out") end; loadpath = "./viz2D_mxpu_out/"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2]
        if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        C_v   = zeros(nx_v, ny_v) # global array for visu
        C_inn = zeros(nx-2, ny-2) # no halo local array for visu
        Xi_g, Yi_g = LinRange(dx+dx/2, Lx-dx-dx/2, nx_v), LinRange(dy+dy/2, Ly-dy-dy/2, ny_v) # inner points only
    end
    t_tic = 0.0; niter = 0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time(); niter = 0 end
        @hide_communication (8, 2) begin
            @parallel compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
            C, C2 = C2, C # pointer swap
            update_halo!(C)
        end
        niter += 1
        # Visualize
        if do_visu && (it % nout == 0)
            C_inn .= C[2:end-1,2:end-1]; gather!(C_inn, C_v)
            if (me==0)
                opts = (aspect_ratio=1, xlims=(Xi_g[1], Xi_g[end]), ylims=(Yi_g[1], Yi_g[end]), clims=(0.0, 0.25), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
                heatmap(Xi_g, Yi_g, Array(C_v)'; opts...); frame(anim)
            end
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = 2/1e9*nx_g()*ny_g()*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                          # Execution time per iteration [s]
    T_eff = A_eff/t_it                           # Effective memory throughput [GB/s]
    if (me==0) @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter) end
    if (do_visu && me==0) gif(anim, "diffusion_2D_mxpu.gif", fps = 5)  end
    finalize_global_grid()
    return
end