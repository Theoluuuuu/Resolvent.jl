# Definition of the function to compute the pipe-flow resolvent matrix at a
# specific Fourier mode (Section 3.2.3 of the report).

struct PipeResolvent{Nr}
    H::Matrix{ComplexF64}
    H_inv::Matrix{ComplexF64}
    M::Matrix{Float64}
    Δ::Matrix{ComplexF64}
    Dr::Matrix{Float64}
    Dr2::Matrix{Float64}
    Rinv::Diagonal{Float64, Vector{Float64}}
    Rinv2::Diagonal{Float64, Vector{Float64}}
    id::Diagonal{Bool, Vector{Bool}}

    function PipeResolvent(Nr::Int, r, Dr, Dr2)
        Z = zeros(Nr, Nr)
        M = [I Z Z;
             Z I Z;
             Z Z I;
             Z Z Z]
        M[Nr,   :] .= 0.0
        M[2*Nr, :] .= 0.0
        M[3*Nr, :] .= 0.0
        new{Nr}(zeros(ComplexF64, 4*Nr, 3*Nr), zeros(ComplexF64, 4*Nr, 4*Nr), M, zeros(ComplexF64, Nr, Nr), Dr, Dr2, Diagonal(1.0 ./ r), Diagonal(1.0 ./ (r.*r)), I(Nr))
    end
end
Base.size(::PipeResolvent{Nr}) where {Nr} = Nr

(f::PipeResolvent)(m, k, n, α, ω, U, dUdr, Re; cz=0.0, cθ=0.0) = f(m, α*k, n*ω - cz*α*k - cθ*m, U, dUdr, Re)

function (f::PipeResolvent{Nr})(m, kz, Ω, U, dUdr, Re) where {Nr}
    # define mode-wise Laplacian
    f.Δ .= f.Dr2 .+ f.Rinv*f.Dr .- (m^2).*f.Rinv2 .- (kz^2).*f.id

    # fill resolvent matrix
    f.H_inv[1:Nr, 1:Nr]                           .= 1im.*Ω.*f.id .+ 1im.*kz.*Diagonal(U) .- (f.Δ .- f.Rinv2)./Re
    f.H_inv[1:Nr, (Nr + 1):(2*Nr)]                .= (2im*m/Re).*f.Rinv2
    f.H_inv[1:Nr, (3*Nr + 1):end]                 .= f.Dr
    f.H_inv[(Nr + 1):(2*Nr), 1:Nr]                .= -(2im*m/Re).*f.Rinv2
    f.H_inv[(Nr + 1):(2*Nr), (Nr + 1):(2*Nr)]     .= 1im.*Ω.*f.id .+ 1im.*kz.*Diagonal(U) .- (f.Δ .- f.Rinv2)./Re
    f.H_inv[(Nr + 1):(2*Nr), (3*Nr + 1):end]      .= 1im.*m.*f.Rinv
    f.H_inv[(2*Nr + 1):(3*Nr), 1:Nr]              .= Diagonal(dUdr)
    f.H_inv[(2*Nr + 1):(3*Nr), (2*Nr + 1):(3*Nr)] .= 1im.*Ω.*f.id .+ 1im.*kz.*Diagonal(U) .- f.Δ./Re
    f.H_inv[(2*Nr + 1):(3*Nr), (3*Nr + 1):end]    .= 1im.*kz.*f.id
    f.H_inv[(3*Nr + 1):end, 1:Nr]                 .= f.Dr .+ f.Rinv
    f.H_inv[(3*Nr + 1):end, (Nr + 1):(2*Nr)]      .= 1im.*m.*f.Rinv
    f.H_inv[(3*Nr + 1):end, (2*Nr + 1):(3*Nr)]    .= 1im.*kz.*f.id

    # apply boundary conditions
    f.H_inv[Nr,   :] .= 0.0; f.H_inv[Nr,   Nr]   = 1.0
    f.H_inv[2*Nr, :] .= 0.0; f.H_inv[2*Nr, 2*Nr] = 1.0
    f.H_inv[3*Nr, :] .= 0.0; f.H_inv[3*Nr, 3*Nr] = 1.0

    # invert resolvent and multiply by influence matrix
    mul!(f.H, inv(f.H_inv), f.M)

    return f.H
end