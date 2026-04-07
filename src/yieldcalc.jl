function logdensities(ξ, M, Zs)
    log_ρs = M * ξ .+ log.(Zs)
    return log_ρs
end
densities(ξ, M, Zs) = exp.(logdensities(ξ, M, Zs))
densities(ϕs, εs, M, Zs; solve_kwargs...) = densities([μs_of_ϕs(ϕs, εs, M, Zs; solve_kwargs...); εs], M, Zs)

function _monomer_densities(ξ, M, ns, Zs)
    return ns' * densities(ξ, M, Zs)
end
monomer_densities(ξ, M, Zs) = _monomer_densities(ξ, M, view(M, :, 1:n_species(M)), Zs)
monomer_densities(ϕs, εs, M, Zs; solve_kwargs...) = monomer_densities([μs_of_ϕs(ϕs, εs, M, Zs; solve_kwargs...); εs], M, Zs)

# function μs_of_ϕs(ϕs, εs, M, Zs; atol=1e-6, rtol=1e-6, maxiters=100_000)
#     nμ = length(ϕs)
#     N = M[:, 1:nμ]
#     B = M[:, nμ+1:end]

#     f!, jac!, jvp!, vjp! = _setup_conversion(ϕs, N, B, Zs)
#     f = NonlinearFunction(f!, jac=jac!, jvp=jvp!, vjp=vjp!)
    
#     init_μs = -1.5 * mean(εs) * ones(nμ)
#     prob = NonlinearProblem(f, init_μs, εs, abstol=atol, reltol=rtol)
#     solution = solve(prob; maxiters)

#     if solution.retcode == ReturnCode.Success
#         return Vector(solution.u)
#     elseif solution.retcode == ReturnCode.Stalled
#         @warn "solution status stalled, proceed with care"
#         return Vector(solution.u)
#     else
#         return fill(Missing, nμ)
#     end
# end

function μs_of_ϕs(ϕs, εs, M, Zs; atol=1e-6, rtol=1e-6, maxiters=1_000_000, infval=10*maximum(εs))
    if any(<(0), ϕs)
        throw(ArgumentError("Particle concentrations cannot be negative."))
    end
    ns = length(Zs)
    nμ = length(ϕs)
    μrange = findall(ϕs .> 0)

    if length(μrange) != nμ
        μforbid = setdiff(1:nμ, μrange)
        strrange = findall(vec(reduce(*, M[:, μforbid] .== 0; dims=2)))
    else
        strrange = 1:ns
    end

    N = M[strrange, 1:nμ]
    B = M[strrange, nμ+1:end]
    Zs = Zs[strrange]

    logNs = [log.(N[N[:, i] .> 0, i]) for i in μrange]
    Ns = [N[N[:, i] .> 0, μrange] for i in μrange]
    Bs = [B[N[:, i] .> 0, :] for i in μrange]
    logZs = [log.(Zs[N[:, i] .> 0]) for i in μrange]

    logϕ(μs, εs, i) = LogExpFunctions.logsumexp(@views logNs[i] .+ logZs[i] .+ Ns[i]*μs .+ Bs[i]*εs)
    f(μs, εs) = [logϕ(μs, εs, i) - log(ϕs[μrange[i]]) for i in eachindex(μrange)] 
    
    init_μs = -1.1 * maximum(εs) * ones(length(μrange))
    prob = NonlinearProblem(f, init_μs, εs, abstol=atol, reltol=rtol)
    solution = solve(prob; maxiters)

    if solution.retcode != ReturnCode.Success && solution.retcode != ReturnCode.Stalled
        @error "solution status $(solution.retcode)"
    end

    μs = -infval * ones(nμ)
    μs[μrange] .= solution.u

    if solution.retcode == ReturnCode.Stalled
        @warn "solution status stalled, proceed with care"
    end

    return μs
end

function logyields(ξ, M, Zs)
    log_ρs = logdensities(ξ, M, Zs)
    log_ρtot = LogExpFunctions.logsumexp(log_ρs)
    return log_ρs .- log_ρtot
end
logyields(ϕs, εs, M, Zs; solve_kwargs...) = logyields([μs_of_ϕs(ϕs, εs, M, Zs; solve_kwargs...); εs], M, Zs)

yields(ξ, M, Zs) = exp.(logyields(ξ, M, Zs))
yields(ϕs, εs, M, Zs; solve_kwargs...) = yields([μs_of_ϕs(ϕs, εs, M, Zs; solve_kwargs...); εs], M, Zs)


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
        jvp!(vJ, v, μs, εs)
        return 
    end
    return f!, jac!, jvp!, vjp!
end

function ∂ρ∂μ(ξ, M, Zs)
    ρs = densities(ξ, M, Zs)
    return ρs .* M
end
function ∂Y∂μ(ξ, M, Zs)
    ρs = densities(ξ, M, Zs)
    Ys = yields(ξ, M, Zs) # just to be safe from numerical issues, recompute
    Σρ = sum(ρs)

    ∂ρs = ∂ρ∂μ(ξ, M, Zs)
    return (∂ρs - Ys .* sum(∂ρs, dims=1)) / Σρ
end
function ∂ρ∂ϕ(ϕs, εs, M, Zs)
    np = length(ϕs)
    N = @view M[:, 1:np]
    B = @view M[:, np+1:end]

    ρs = densities(ϕs, εs, M, Zs)
    ∂ϕ∂μ =  N' * Diagonal(ρs) * N
    ∂ϕ∂ε =  N' * Diagonal(ρs) * B

    ∂μ∂ϕ = inv(∂ϕ∂μ)
    ∂μ∂ε = -∂μ∂ϕ * ∂ϕ∂ε

    ∂ρ∂ϕ = ρs .* N * ∂μ∂ϕ
    ∂ρ∂ε = ρs .* (N * ∂μ∂ε + B)

    return hcat(∂ρ∂ϕ, ∂ρ∂ε)
end

function ∂Y∂ϕ(ϕs, εs, M, Zs)
    ρs = densities(ϕs, εs, M, Zs)
    Ys = yields(ϕs, εs, M, Zs) # just to be safe from numerical issues, recompute
    Σρ = sum(ρs)

    ∂ρs = ∂ρ∂ϕ(ϕs, εs, M, Zs)
    return (∂ρs - Ys .* sum(∂ρs, dims=1)) / Σρ
end

function ∂ϕ∂μ(ξ, M, Zs)
    np = n_species(M)
    N = @view M[:, 1:np]
    B = @view M[:, np+1:end]

    ρs = densities(ξ, M, Zs)
    ∂ϕ∂μ =  N' * Diagonal(ρs) * N
    ∂ϕ∂ε =  N' * Diagonal(ρs) * B
    return hcat(∂ϕ∂μ, ∂ϕ∂ε)
end