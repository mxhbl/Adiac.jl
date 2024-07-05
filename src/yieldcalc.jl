function logdensities(ξ, M, Zs)
    log_ρs = M * ξ .+ log.(Zs)
    return log_ρs
end
densities(ξ, M, Zs) = exp.(logdensities(ξ, M, Zs))
densities(ϕs, εs, M, Zs) = densities([μs_of_ϕs(ϕs, εs, M, Zs); εs], M, Zs)

function monomer_densities(ξ, M, nμ, Zs)
    ns = M[:, 1:nμ]
    return ns' * densities(ξ, M, Zs)
end
monomer_densities(ξ, M, Zs) = monomer_densities(ξ, M, n_species(M), Zs)

function μs_of_ϕs(ϕs, εs, M, Zs; ε_abs=1e-6, ε_rel=1e-6)
    nμ = length(ϕs)
    f(u, args...) = monomer_densities([u; εs], M, nμ, Zs) - ϕs
    init_μs = -1.5 * mean(εs) * ones(nμ)
    prob = NonlinearProblem(f, init_μs, zeros(1), abstol=ε_abs, reltol=ε_rel)
    solution = solve(prob)

    if solution.retcode == ReturnCode.Success
        return Vector(solution.u)
    else
        return fill(Missing, nμ)
    end
end

function logyields(ξ, M, Zs)
    log_ρs = logdensities(ξ, M, Zs)
    log_ρtot = LogExpFunctions.logsumexp(log_ρs)
    return log_ρs .- log_ρtot
end
yields(ξ, M, Zs) = exp.(logyields(ξ, M, Zs))
yields(ϕs, εs, M, Zs) = yields([μs_of_ϕs(ϕs, εs, M, Zs); εs], M, Zs)
