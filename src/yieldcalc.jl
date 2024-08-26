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
    N = M[:, 1:nμ]
    B = M[:, nμ+1:end]

    f!, jac!, jvp!, vjp! = _setup_conversion(ϕs, N, B, Zs)
    f = NonlinearFunction(f!, jac=jac!, jvp=jvp!, vjp=vjp!)
    
    init_μs = -1.5 * mean(εs) * ones(nμ)
    prob = NonlinearProblem(f, init_μs, εs, abstol=atol, reltol=rtol)
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
yields(ϕs, εs, M, Zs; atol=1e-6, rtol=1e-6) = yields([μs_of_ϕs(ϕs, εs, M, Zs; atol=atol, rtol=rtol); εs], M, Zs)


function _setup_conversion(ϕs_target, N, B, Zs)
    function f!(dϕs, μs, εs)
        dϕs .= N' * (exp.(N * μs + B * εs) .* Zs) - ϕs_target
        return 
    end
    function jac!(J, μs, εs)
        J .=  N' * Diagonal(exp.(N * μs + B * εs) .* Zs) * N
        return 
    end
    function jvp!(Jv, v, μs, εs)
        Jv .=  N' * Diagonal(exp.(N * μs + B * εs) .* Zs) * (N * v)
        return 
    end
    function vjp!(vJ, v, μs, εs)
        vJ .= (N * v)' * Diagonal(exp.(N * μs + B * εs) .* Zs) * N
        return 
    end
    return f!, jac!, jvp!, vjp!
end