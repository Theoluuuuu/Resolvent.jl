# Utility method to generate pipe-flow response modes for a set of parameters.
# `even` and `odd` are the radial derivative pairs (Dr, Dr2) mirrored at the axis
# with even and odd parity; PipeResolvent picks the parity of each component.
# The (m, k) = (0, 0) modes carry no flux unless constant_flux = false. Only the
# modes with nonzero singular values are kept (9-month report, Section 3.3.3);
# they span the velocities that satisfy continuity and the wall and flux
# conditions, 2N - 3 of them for a whole block and N - 2 for the smallest
# subsystem kept by the symmetry, so M must not exceed that number.
#
# The resolvents of (m, k, n) and (-m, -k, -n) are complex conjugates. Only m ≥ 0
# is stored, so at m = 0 the modes of (0, k, n) with k < 0, or k = 0 and n < 0,
# are the conjugates of those of (0, -k, -n), and those of (0, 0, 0) are real
# (9-month report, Section 3.3.3).
#
# With `shift_reflect` (the default) the modes form the basis of the search
# restricted to the shift-and-reflect symmetry u(r, θ, z) = S u = (u_r, -u_θ,
# u_z)(r, -θ, z - L/2), which requires cθ = 0 (9-month report, Section 3.3.4).
# Writing R for the reversal of the azimuthal component:
#   - m > 0: the blocks (m, k, n) and (m, -k, -n) are a pair. The modes of the
#     first of the pair with k > 0, or k = 0 and n > 0, come from the singular
#     value decomposition; those of the other are conj((-1)^k R ψ), so the
#     coefficients satisfy a_(m,-k,-n) = conj(a_(m,k,n)) and the pair is one
#     complex variable. For (m, 0, 0) the phase of each mode is chosen so that
#     ψ = conj(R ψ), and its coefficient is real.
#   - m = 0, k ≠ 0: the symmetry keeps u_r and u_z for even k and u_θ for odd
#     k, which decouple at m = 0; the modes are those of that subsystem, zero in
#     the other components.
#   - (m, k) = (0, 0): the symmetry removes u_θ, and continuity with axis
#     regularity gives u_r = 0; the modes are those of the axial component, from
#     the decomposition at each n.

export generate_pipe_modes!, generate_pipe_modes

# leading M singular triplets of the block idx × idx of the velocity rows of H,
# in the r dr weights W
function _pipe_svd(H, W, idx, M)
    w = sqrt.(W[idx])
    F = svd((w .* H[idx, idx]) ./ transpose(w))
    return F.U[:, 1:M] ./ w, F.S[1:M], F.V[:, 1:M] ./ w
end

_reflect(ψ, N) = vcat(ψ[1:N, :], -ψ[(N + 1):(2*N), :], ψ[(2*N + 1):(3*N), :])

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
                   shift_reflect::Bool = true,
                         verbose::Bool = true) where {T}
    # unpack useful variables from inputs
    N3, M, Nm, Nk, Nn = size(Ψ)
    N = N3 ÷ 3
    W = repeat(ws, 3)
    index(nm, nk, nn) = (nm + 1, nk >= 0 ? nk + 1 : Nk + nk + 1, nn >= 0 ? nn + 1 : Nn + nn + 1)
    isodd(Nk) && isodd(Nn) || throw(ArgumentError("the numbers of k and n must be odd, so that every block has its partner"))
    shift_reflect && cθ != 0 && throw(ArgumentError("the shift-and-reflect symmetry requires cθ = 0"))
    M <= (shift_reflect ? N - 2 : 2*N - 3) || throw(ArgumentError("M = $M exceeds the number of nonzero singular values"))

    # the blocks whose modes come from a decomposition: at m = 0 the first of
    # each conjugate pair and (0, 0, 0), and with the symmetry the same at m > 0
    first_of_pair(nk, nn) = nk > 0 || (nk == 0 && nn > 0)
    decomposed(nm, nk, nn) = first_of_pair(nk, nn) || (nk == 0 && nn == 0) || (nm > 0 && !shift_reflect)

    # initialise pipe resolvent operator
    H = PipeResolvent(N, r, ws, even, odd; constant_flux=constant_flux)

    verbose && println("m: 0:$(Nm-1), k: $(-(Nk >> 1)):$(Nk >> 1), n: $(-(Nn >> 1)):$(Nn >> 1)")
    blocks = [(nm, nk, nn) for nn in -(Nn >> 1):(Nn >> 1), nk in -(Nk >> 1):(Nk >> 1), nm in 0:Nm-1]

    # first pass: loop over the decomposed blocks computing response modes
    for (nm, nk, nn) in blocks
        decomposed(nm, nk, nn) || continue
        verbose && print("m=$nm, k=$nk, n=$nn                    \r"); flush(stdout)
        Hk = H(nm, nk, nn, α, ω, U, dUdr, Re; cz=cz, cθ=cθ)
        if nm == 0 && nk == 0 && nn == 0
            # the operator is real here, and so are its modes
            idx = shift_reflect ? collect((2*N + 1):(3*N)) : collect(1:(3*N))
            ψ, _, _ = _pipe_svd(real.(Hk), W, idx, M)
            ψ = ψ .* transpose([sign(c[argmax(abs.(c))]) for c in eachcol(ψ)])
        elseif nm == 0 && shift_reflect
            idx = nk == 0 ? collect((2*N + 1):(3*N)) :
                  iseven(nk) ? vcat(1:N, (2*N + 1):(3*N)) : collect((N + 1):(2*N))
            ψ, _, _ = _pipe_svd(Hk, W, idx, M)
        else
            idx = collect(1:(3*N))
            ψ = svd(Hk, ws, M).U
            if shift_reflect && nk == 0 && nn == 0
                # phase making ψ = conj(R ψ): if conj(R ψ) = e^{iχ} ψ, multiply by e^{iχ/2}
                χ = angle.(vec(sum(W .* conj.(ψ) .* conj.(_reflect(ψ, N)); dims=1)))
                ψ = ψ .* transpose(cis.(χ ./ 2))
            end
        end
        Ψ[:, :, index(nm, nk, nn)...] .= 0
        Ψ[idx, :, index(nm, nk, nn)...] .= ψ
    end

    # second pass: the other block of each pair, from the first
    for (nm, nk, nn) in blocks
        decomposed(nm, nk, nn) && continue
        if nm == 0
            Ψ[:, :, index(nm, nk, nn)...] .= conj.(Ψ[:, :, index(0, -nk, -nn)...])
        else
            Ψ[:, :, index(nm, nk, nn)...] .= (-1)^nk .* conj.(_reflect(Ψ[:, :, index(nm, -nk, -nn)...], N))
        end
    end

    return Ψ
end

generate_pipe_modes(S, M, Re, r, even, odd, ws, α, ω, ::Type{T}=Float64;
                       U = T(2) .* (one(T) .- T.(r).^2),
                    dUdr = T(-4) .* T.(r),
                      cz = zero(T),
                      cθ = zero(T),
           constant_flux = true,
           shift_reflect = true,
                 verbose = true) where {T} =
    generate_pipe_modes!(zeros(Complex{T}, 3*S[1], M, (S[2] >> 1) + 1, S[3], S[4]),
                         T(Re), T.(r), Tuple(T.(D) for D in even), Tuple(T.(D) for D in odd), T.(ws), T(α), T(ω);
                         U=T.(U), dUdr=T.(dUdr), cz=T(cz), cθ=T(cθ), constant_flux=constant_flux,
                         shift_reflect=shift_reflect, verbose=verbose)
