# julia -O3 --check-bounds=no -t 4 ./scripts-part1/diffusion3D_visualize.jl

# Initialize Globals
const USE_GPU = true
const BENCHMARK = false
const VISUALIZE = true

# Include main script
include("./diffusion3D_multixpu.jl")

# Run the simulation
diffusion_3D(256)