using SparseArrays, ArnoldiMethod

function is_related(ΔM)
    ds = @view ΔM[ΔM .!= 0]
    return (abs(ds[1]) == 1 && length(ds) == 2 && all(ds .== ds[1])) ? ds[1] : 0
end

function monoadd_kinetics(M, ξ)
    nstr, _ = size(M)
    T = spzeros(eltype(ξ), nstr, nstr)

    for i in axes(T, 1), j in i+1:size(T, 2)
        ΔM = M[i, :] - M[j, :]
        relation = is_related(ΔM)

        if relation == 0
            continue
        end

        # TODO: optimize
        @views μ, ε = ξ[ΔM .== relation]
        if relation == 1
            T[i, j] = exp(μ)
            T[j, i] = exp(-ε)
        else
            T[i, j] = exp(-ε)
            T[j, i] = exp(μ)
        end
    end

    T -= spdiagm(vec(sum(T, dims=1)))
    return T
end

function stat_dist(T; thresh=1e-12)
    decomp, _ = partialschur(T, nev=2, which=:LR)
    @assert abs.(decomp.eigenvalues[1]) < thresh
    @assert abs.(decomp.eigenvalues[2]) > thresh

    πvec = decomp.Q[:, 1]
    return πvec / sum(πvec)
end

function massaction_kinetics(M)
end