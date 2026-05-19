module ResolventAnalysis

using LinearAlgebra, TSVD

export Resolvent, PipeResolvent, svd
export DivideAndConquer, QRIteration, Lanczos, Adaptive
export slab_modes, load_modes, load_modes!, save_modes

include("resolvent.jl")
include("pipe_resolvent.jl")
include("cholesky.jl")
include("svd.jl")
include("generate_modes.jl")
include("generate_pipe_modes.jl")
include("load.jl")

end
