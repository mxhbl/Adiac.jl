using SparseArrays, ArnoldiMethod

function n_species(M::AbstractMatrix)
    nμ = 0
    while sum(M[nμ + 1, :]) == 1
        nμ += 1
    end
    return nμ
end

function monoadd_kinetics(M, ξ, ψ)
    nstr, _ = size(M)
    nμ = n_species(M)

    T = spzeros(eltype(ξ), nstr, nstr)

    for i in axes(T, 1), j in i+1:size(T, 2)
        ΔM = M[i, :] - M[j, :]
        δμ = ΔM[1:nμ]
        δε = ΔM[nμ+1:end]

        # if not all δε have the same sign, continue
        if sum(abs.(δε)) != abs(sum(δε))
            continue
        end

        if all(δμ .>= 0) && sum(δμ) == 1
            T[i, j] = exp(ξ[1:nμ]' * abs.(δμ))
            T[j, i] = exp(-ξ[nμ+1:end]' * abs.(δε))
        elseif all(δμ .<= 0) && sum(δμ) == -1
            T[j, i] = exp(ξ[1:nμ]' * abs.(δμ))
            T[i, j] = exp(-ξ[nμ+1:end]' * abs.(δε))
        end
    end

    T -= spdiagm(vec(sum(T, dims=1)))
    return T
end
monoadd_kinetics(M, ξ) = monoadd_kinetics(M, ξ, zeros(eltype(ξ), size(M, 1)))

function stat_dist(T; thresh=1e-8)
    decomp, _ = partialschur(T, nev=2, which=LR())
    @assert abs.(decomp.eigenvalues[1]) < thresh
    @assert abs.(decomp.eigenvalues[2]) > thresh

    πvec = decomp.Q[:, 1]
    return πvec / sum(πvec)
end

function massaction_kinetics(M)
end

#TODO move this to the appropriate place
function moment_sum(p::Polyform, χ)
    χ_tot = zeros(eltype(χ), 2)
    for (species, ψ) in zip(p.species, p.ψs)
        χ_tot += Adiac.rotate(χ[:, species], ψ.θ)
    end
    return χ_tot
end