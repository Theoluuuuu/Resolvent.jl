# Definition of the function to compute the pipe-flow resolvent matrix at a
# specific Fourier mode, in the structure of the channel `Resolvent`: the
# velocity–total-pressure system with the momentum equations at the wall replaced
# by the no-slip conditions (the radial one by the pressure gauge at
# (m, k) = (0, 0)), inverted and multiplied by the influence matrix (9-month
# report, Section 3.3.2 and Appendix B.2).
#
# Sign convention: that of this package, a mode exp(i(mθ + kz z + Ω t)) with
# ∂_t -> +iΩ, Ω being its frequency in the frame fixed to the pipe. For the
# integer wavenumber k, kz = αk, and the frequency nω in the frame drifting at
# (cz, cθ), Ω = nω - cz αk - cθ m. The report writes the mode as
# exp(i(k_θθ' + αk_z z' - 2πk_t t/T)) in the drifting coordinates θ' = θ - cθ t,
# z' = z - cz t (Section 3.3.2), i.e. exp(i(k_θθ + αk_z z - ω_k t)) in the frame
# fixed to the pipe, so m = k_θ, k = k_z, ω = 2π/T, n = -k_t and Ω = -ω_k.
#
# Unknowns stack (u_r, u_θ, u_z, p) on the radial grid r (wall at r[end] = 1, no
# point on the axis), then the pressure drop β, so that the total pressure is
# P = p - β z; ws are the weights of ∫ f r dr. Axis regularity takes the parity
# of m for u_z and p and that of m + 1 for u_r and u_θ (Lewis & Bellan 1990), so
# the radial derivative matrices are supplied for both parities:
# `even = (Dr, Dr2)` mirrored with even and `odd = (Dr, Dr2)` with odd parity at
# the axis.
#
# The pressure drop enters only at (m, k) = (0, 0) with `constant_flux`, as the
# multiplier of the zero-flux condition ∫ u_z r dr = 0, which takes the last
# row; elsewhere its row and column are those of β = 0. At (m, k) = (0, 0) the
# pressure enters only the radial momentum equation, so its additive constant is
# free, and continuity with axis regularity gives u_r = 0, so u_r(1) = 0 holds
# without being imposed. That wall row takes the pressure gauge ∫ p r dr = 0
# instead, and the system is inverted as at every other (m, k).

struct PipeResolvent{Nr}
    H::Matrix{ComplexF64}
    H_inv::Matrix{ComplexF64}
    M::Matrix{Float64}
    Δ::Matrix{Float64}
    vΔ::Matrix{Float64}
    r::Vector{Float64}
    ws::Vector{Float64}
    even::NTuple{2, Matrix{Float64}}
    odd::NTuple{2, Matrix{Float64}}
    Rinv::Diagonal{Float64, Vector{Float64}}
    Rinv2::Diagonal{Float64, Vector{Float64}}
    id::Diagonal{Bool, Vector{Bool}}
    constant_flux::Bool

    function PipeResolvent(Nr::Int, r, ws, even, odd; constant_flux::Bool=true)
        length(r) == length(ws) == Nr || throw(ArgumentError("r and ws must have Nr entries"))
        Z = zeros(Nr, Nr)
        M = [I Z Z;
             Z I Z;
             Z Z I;
             Z Z Z;
             zeros(1, 3*Nr)]
        M[Nr,   :] .= 0.0
        M[2*Nr, :] .= 0.0
        M[3*Nr, :] .= 0.0
        new{Nr}(zeros(ComplexF64, 4*Nr, 3*Nr), zeros(ComplexF64, 4*Nr + 1, 4*Nr + 1), M,
                zeros(Nr, Nr), zeros(Nr, Nr), collect(Float64, r), collect(Float64, ws),
                Tuple(Matrix{Float64}.(even)), Tuple(Matrix{Float64}.(odd)),
                Diagonal(1.0 ./ r), Diagonal(1.0 ./ (r.*r)), I(Nr), constant_flux)
    end
end
Base.size(::PipeResolvent{Nr}) where {Nr} = Nr

(f::PipeResolvent)(m, k, n, α, ω, U, dUdr, Re; cz=0.0, cθ=0.0) = f(m, α*k, n*ω - cz*α*k - cθ*m, U, dUdr, Re)

_parity(f::PipeResolvent, p) = iseven(p) ? f.even : f.odd

function (f::PipeResolvent{Nr})(m, kz, Ω, U, dUdr, Re) where {Nr}
    Dr, Dr2 = _parity(f, m)          # u_z, p
    vDr, vDr2 = _parity(f, m + 1)    # u_r, u_θ

    # define mode-wise Laplacians
    f.Δ  .= Dr2  .+ f.Rinv*Dr  .- (m^2).*f.Rinv2 .- (kz^2).*f.id
    f.vΔ .= vDr2 .+ f.Rinv*vDr .- (m^2).*f.Rinv2 .- (kz^2).*f.id

    # fill resolvent matrix
    f.H_inv .= 0.0
    f.H_inv[1:Nr, 1:Nr]                           .= 1im.*Ω.*f.id .+ 1im.*kz.*Diagonal(U) .- (f.vΔ .- f.Rinv2)./Re
    f.H_inv[1:Nr, (Nr + 1):(2*Nr)]                .= (2im*m/Re).*f.Rinv2
    f.H_inv[1:Nr, (3*Nr + 1):(4*Nr)]              .= Dr
    f.H_inv[(Nr + 1):(2*Nr), 1:Nr]                .= -(2im*m/Re).*f.Rinv2
    f.H_inv[(Nr + 1):(2*Nr), (Nr + 1):(2*Nr)]     .= 1im.*Ω.*f.id .+ 1im.*kz.*Diagonal(U) .- (f.vΔ .- f.Rinv2)./Re
    f.H_inv[(Nr + 1):(2*Nr), (3*Nr + 1):(4*Nr)]   .= 1im.*m.*f.Rinv
    f.H_inv[(2*Nr + 1):(3*Nr), 1:Nr]              .= Diagonal(dUdr)
    f.H_inv[(2*Nr + 1):(3*Nr), (2*Nr + 1):(3*Nr)] .= 1im.*Ω.*f.id .+ 1im.*kz.*Diagonal(U) .- f.Δ./Re
    f.H_inv[(2*Nr + 1):(3*Nr), (3*Nr + 1):(4*Nr)] .= 1im.*kz.*f.id
    f.H_inv[(3*Nr + 1):(4*Nr), 1:Nr]              .= vDr .+ f.Rinv
    f.H_inv[(3*Nr + 1):(4*Nr), (Nr + 1):(2*Nr)]   .= 1im.*m.*f.Rinv
    f.H_inv[(3*Nr + 1):(4*Nr), (2*Nr + 1):(3*Nr)] .= 1im.*kz.*f.id

    # pressure drop and zero-flux condition at (m, k) = (0, 0), β = 0 elsewhere
    if f.constant_flux && m == 0 && kz == 0
        f.H_inv[(2*Nr + 1):(3*Nr), end] .= -1.0
        f.H_inv[end, (2*Nr + 1):(3*Nr)] .= f.ws
    else
        f.H_inv[end, end] = 1.0
    end

    # apply boundary conditions, with the pressure gauge in place of u_r(1) = 0
    # at (m, k) = (0, 0)
    f.H_inv[Nr,   :] .= 0.0
    if m == 0 && kz == 0
        f.H_inv[Nr, (3*Nr + 1):(4*Nr)] .= f.ws
    else
        f.H_inv[Nr, Nr] = 1.0
    end
    f.H_inv[2*Nr, :] .= 0.0; f.H_inv[2*Nr, 2*Nr] = 1.0
    f.H_inv[3*Nr, :] .= 0.0; f.H_inv[3*Nr, 3*Nr] = 1.0

    # invert resolvent and multiply by influence matrix, dropping the pressure drop;
    # as in Resolvent, H keeps the pressure rows, and svd(H, ws, nvals) uses only
    # the velocity rows
    mul!(f.H, view(inv(f.H_inv), 1:(4*Nr), :), f.M)

    return f.H
end
