# Utility method to generate pipe-flow response modes for a set of parameters.
# `even` and `odd` are the radial derivative pairs (Dr, Dr2) mirrored at the axis
# with even and odd parity; PipeResolvent picks the parity of each component.
# The (m, k) = (0, 0) blocks carry no flux unless constant_flux = false.

export generate_pipe_modes!, generate_pipe_modes

function generate_pipe_modes!(Ψ::Array{Complex{T}, 5},
                              Re::T,
                               r::Vector{T},
                            even::NTuple{2, AbstractMatrix{T}},
                             odd::NTuple{2, AbstractMatrix{T}},
                              ws::Vector{T},
                               α::T,
                               ω::T;
                               U::Vector{T} = T(2) .* (one(T) .- r.^2),
                            dUdr::Vector{T} = T(-4) .* r,
                              cz::T = zero(T),
                              cθ::T = zero(T),
                   constant_flux::Bool = true,
                         verbose::Bool = true) where {T}
    # unpack useful variables from inputs
    N, M, Nm, Nk, Nn = size(Ψ)

    # initialise pipe resolvent operator
    H = PipeResolvent(N÷3, r, ws, even, odd; constant_flux=constant_flux)

    # loop over (m, k, n) computing response modes
    verbose && println("m: 0:$(Nm-1), k: $(-(Nk >> 1)):$(Nk >> 1), n: $(-(Nn >> 1)):$(Nn >> 1)")
    for nn in -(Nn >> 1):(Nn >> 1), nk in -(Nk >> 1):(Nk >> 1), nm in 0:Nm-1
        verbose && print("m=$nm, k=$nk, n=$nn                    \r"); flush(stdout)
        _nm = nm + 1
        _nk = nk >= 0 ? nk + 1 : Nk + nk + 1
        _nn = nn >= 0 ? nn + 1 : Nn + nn + 1
        Ψ[:, :, _nm, _nk, _nn] .= svd(H(nm, nk, nn, α, ω, U, dUdr, Re; cz=cz, cθ=cθ), ws, M).U
    end

    return Ψ
end

generate_pipe_modes(S, M, Re, r, even, odd, ws, α, ω, ::Type{T}=Float64;
                       U = T(2) .* (one(T) .- T.(r).^2),
                    dUdr = T(-4) .* T.(r),
                      cz = zero(T),
                      cθ = zero(T),
           constant_flux = true,
                 verbose = true) where {T} =
    generate_pipe_modes!(zeros(Complex{T}, 3*S[1], M, (S[2] >> 1) + 1, S[3], S[4]),
                         T(Re), T.(r), Tuple(T.(D) for D in even), Tuple(T.(D) for D in odd), T.(ws), T(α), T(ω);
                         U=T.(U), dUdr=T.(dUdr), cz=T(cz), cθ=T(cθ), constant_flux=constant_flux, verbose=verbose)