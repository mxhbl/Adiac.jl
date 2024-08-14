using SparseArrays, ArnoldiMethod

function monoadd_kinetics(M, ξ, ψ=nothing; vacuum=false)
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

    # TODO optimize
    if vacuum
        TT = spzeros(Float64, nstr+1, nstr+1)
        TT[2:nμ+1, 1] = exp.(ξ[1:nμ])
        TT[1, 2:nμ+1] = exp.(-ξ[1:nμ])
        TT[2:end, 2:end] .= T
        T = TT

        if !isnothing(ψ)
            T[1, nμ+2:end] = ψ
        end
    end

    T -= spdiagm(vec(sum(T, dims=1)))
    return T
end

function stat_dist(T; vacuum=false, thresh=1e-8)
    # decomp, _ = partialschur(T, nev=2, which=LR())
    # @assert abs.(decomp.eigenvalues[1]) < thresh
    # println(abs.(decomp.eigenvalues))
    # πvec = decomp.Q[:, 1]

    λs, V = eigen(Matrix(T)) # TODO get partialschur to work
    @assert abs.(λs[end]) < thresh
    πvec = V[:, end]
    println(abs.(λs[end-1:end]))

    if vacuum
        πvec = πvec[2:end]
    end

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
function breakup_rates(strs::AbstractVector{<:Polyform}, χ)
    return [norm(x) for (p, x) in zip(strs, moment_sum.(strs, Ref(χ))) if size(p) > 1]
end


M = [1 0 0;
     0 1 0;
     1 1 1;
     1 2 2;
     1 3 3]