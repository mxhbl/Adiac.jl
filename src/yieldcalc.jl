function logdensities(ξ, M, Zs)
    log_ρs = M * ξ .+ log.(Zs)
    return log_ρs
end
densities(ξ, M, Zs) = exp.(logdensities(ξ, M, Zs))
densities(ϕs, εs, M, Zs; atol=1e-6, rtol=1e-6) = densities([μs_of_ϕs(ϕs, εs, M, Zs; atol=atol, rtol=rtol); εs], M, Zs)

function _monomer_densities(ξ, M, ns, Zs)
    return ns' * densities(ξ, M, Zs)
end
monomer_densities(ξ, M, Zs) = _monomer_densities(ξ, M, view(M, :, 1:n_species(M)), Zs)

function μs_of_ϕs(ϕs, εs, M, Zs; atol=1e-6, rtol=1e-6)
    nμ = length(ϕs)
    ns = M[:, 1:n_species(M)]
    f(u, args...) = _monomer_densities([u; εs], M, ns, Zs) - ϕs
    
    init_μs = -1.5 * mean(εs) * ones(nμ)
    prob = NonlinearProblem(f, init_μs, zeros(1), abstol=atol, reltol=rtol)
    solution = solve(prob, NewtonRaphson())

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
yields(ϕs, εs, M, Zs; atol=1e-6, rtol=1e-6) = yields([μs_of_ϕs(ϕs, εs, M, Zs; atol=atol, rtol=rtol); εs], M, Zs)
