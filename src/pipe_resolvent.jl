# Pipe-flow resolvent operator at one Fourier mode, in the Galerkin form of the
# 9-month report (Section 3.3.2 and Appendix B) and of PipeResolventModes
# (src/resolvent.jl): the velocity-pressure system restricted to the admissible
# velocities.
#
# Sign convention: that of this package, a mode exp(i(mθ + k z + Ω t)) with
# ∂_t -> +iΩ, where in the frame drifting at (cz, cθ) Ω = nω - cz k - cθ m. The
# report writes exp(i(k_θθ + αk_z z) - iω_k t) instead, so Ω = -ω_k: the same
# operator at the opposite sign of the frequency.
#
# Velocities stack (u_r, u_θ, u_z) on the radial grid r (wall at r[end] = 1, no
# point on the axis), with weights ws for ∫ f r dr. Axis regularity takes the
# parity of m for u_z and p and that of m + 1 for u_r and u_θ (Lupi et al.
# 2024, section 3.1), so the radial derivative matrices are supplied for both
# parities: `even = (Dr, Dr2)` mirrored with even and `odd = (Dr, Dr2)` with odd
# parity at the axis.
#
# The admissible velocities satisfy the discrete continuity equation, vanish at
# the wall and, for (m, k) = (0, 0) with `constant_flux`, carry no flux,
# ∫ u_z r dr = 0: the constant-flux condition of a flow driven at fixed mass
# flux, whose pressure drop is the multiplier of that condition. For the other
# blocks the flux vanishes by continuity. With Z a basis of these velocities,
# orthonormal in the weights W = [ws; ws; ws], and A(Ω) = iΩ I - L the velocity
# block of the system, Z^H W A(Ω) Z = M0 + iΩ I, and the resolvent operator is
# H(Ω) = Z (M0 + iΩ I)^{-1} Z^H W. Forcing and response modes then lie in the
# admissible space and the total pressure gradient, which is orthogonal to it,
# drops out; the pressure is never solved for, so the (0, 0) blocks, where it is
# fixed only up to a constant, need no gauge.

struct PipeResolvent{Nr}
    H::Matrix{ComplexF64}                       # 4Nr × 3Nr: velocity rows, then zero pressure rows
    r::Vector{Float64}
    ws::Vector{Float64}
    even::NTuple{2, Matrix{Float64}}
    odd::NTuple{2, Matrix{Float64}}
    constant_flux::Bool
    bases::Dict{Tuple{Int, Float64}, Matrix{ComplexF64}}   # Z for each (m, k)

    function PipeResolvent(Nr::Int, r, ws, even, odd; constant_flux::Bool=true)
        length(r) == length(ws) == Nr || throw(ArgumentError("r and ws must have Nr entries"))
        new{Nr}(zeros(ComplexF64, 4*Nr, 3*Nr), collect(Float64, r), collect(Float64, ws),
                Tuple(Matrix{Float64}.(even)), Tuple(Matrix{Float64}.(odd)), constant_flux,
                Dict{Tuple{Int, Float64}, Matrix{ComplexF64}}())
    end
end
Base.size(::PipeResolvent{Nr}) where {Nr} = Nr

(f::PipeResolvent)(m, k, n, α, ω, U, dUdr, Re; cz=0.0, cθ=0.0) = f(m, α*k, n*ω - cz*α*k - cθ*m, U, dUdr, Re)

_parity(f::PipeResolvent, p) = iseven(p) ? f.even : f.odd

# velocity block A(Ω = 0) = -L and divergence rows at (m, k)
function _pipe_blocks(f::PipeResolvent{Nr}, m, kz, U, dUdr, Re) where {Nr}
    Dr, Dr2 = _parity(f, m)          # u_z, p
    vDr, vDr2 = _parity(f, m + 1)    # u_r, u_θ
    Rinv, Rinv2, id = Diagonal(1 ./ f.r), Diagonal(1 ./ f.r .^ 2), Matrix{Float64}(I, Nr, Nr)
    Δ = Dr2 .+ Rinv*Dr .- (m^2).*Rinv2 .- (kz^2).*id
    vΔ = vDr2 .+ Rinv*vDr .- (m^2).*Rinv2 .- (kz^2).*id
    adv = 1im*kz .* Diagonal(U)
    Z = zeros(ComplexF64, Nr, Nr)
    A = [adv .- (vΔ .- Rinv2)./Re      (2im*m/Re).*Rinv2               Z
         -(2im*m/Re).*Rinv2            adv .- (vΔ .- Rinv2)./Re        Z
         ComplexF64.(Diagonal(dUdr))   Z                               adv .- Δ./Re]
    Div = [vDr .+ Rinv   1im*m .* Rinv   1im*kz .* id]
    return A, Div
end

# basis of the admissible velocities, orthonormal in W
function _admissible_basis(f::PipeResolvent{Nr}, m, kz, Div) where {Nr}
    get!(f.bases, (Int(m), Float64(kz))) do
        wall = zeros(3, 3Nr)
        wall[1, Nr] = wall[2, 2Nr] = wall[3, 3Nr] = 1
        rows = [Div; wall]
        if f.constant_flux && m == 0 && kz == 0
            rows = [rows; zeros(1, 2Nr) f.ws']
        end
        C = svd(rows; full=true)
        Zb = C.V[:, count(>(1e-10*C.S[1]), C.S)+1:end]
        W = repeat(f.ws, 3)
        Zb / cholesky(Hermitian(Zb' * (W .* Zb))).U
    end
end

function (f::PipeResolvent{Nr})(m, kz, Ω, U, dUdr, Re) where {Nr}
    A, Div = _pipe_blocks(f, m, kz, U, dUdr, Re)
    Zb = _admissible_basis(f, m, kz, Div)
    W = repeat(f.ws, 3)
    M0 = Zb' * (W .* (A * Zb))
    # H = Z (M0 + iΩ I)^{-1} Z^H W, stored with zero pressure rows for svd(H, ws, nvals)
    f.H[1:3Nr, :] .= Zb * ((M0 + 1im*Ω*I) \ (Zb' .* W'))
    f.H[3Nr+1:end, :] .= 0
    return f.H
end
