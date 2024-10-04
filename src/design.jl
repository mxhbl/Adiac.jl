function preprocess_optimization(M, idxs)
    """Removes all structures that contain bonds or particles not present in the desired structures"""
    n_structs, n_pars = size(M)

    D = M[idxs, :]
    D_counts = sum(D; dims=1)

    element_mask = vec(D_counts .!= 0)

    if all(element_mask)
        return ones(Bool, n_structs), ones(Bool, n_pars), idxs
    end

    structure_mask = vec((M * .!element_mask) .== 0)
    new_idxs = [sum(structure_mask[1:i]) for i in idxs]

    return structure_mask, element_mask, new_idxs
end

function linear_design(M, idxs; preprocess=true, refine_undesignable=true, atol=1e-6, rtol=1e-6, infval=100)
    nμ = n_species(M)

    if isa(idxs, Number)
        idxs = [idxs]
    end

    if preprocess
        structure_mask, element_mask, new_idxs = preprocess_optimization(M, idxs)

        M = M[structure_mask, element_mask]
        idxs = new_idxs

        missing_pars = findall(.!element_mask)
        if !isempty(missing_pars)
            nμ = nμ - sum(missing_pars .<= nμ)
        end
    end

    _, npars = size(M)

    if npars > 1
        x = Variable(npars)
        A = M[setdiff(1:end, idxs), :]
        B = M[idxs, :]

        problem = minimize(maximum(A * x), B * x == 0,
                           sum(x[(nμ + 1):end]) - sum(x[1:nμ]) == npars)

        Convex.solve!(problem,
               Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "verbose" => 0,
                                                  "eps_abs" => atol, "eps_rel" => rtol))
        xi_hat = vec(x.value)
        residual = problem.optval
    else
        xi_hat = [0.0]
        residual = -Inf
    end

    structures = findall(isapprox.(M * xi_hat, 0.0; atol=atol))
    chimeras = setdiff(structures, idxs)
    if refine_undesignable && !isempty(chimeras)
        @warn "Target set is not designable, optimizing for minimal enclosing designable set..."
        extended_idxs = union(idxs, chimeras)
        xi_hat, residual = linear_design(M, extended_idxs; preprocess=false)
    end

    if preprocess
        for mi in missing_pars
            insert!(xi_hat, mi, -Inf)
        end
    end

    return infapprox(xi_hat, infval), residual
end

function convex_design(M, i; max_ε=1, max_ϕ=1, σs=nothing, preprocess=true, max_steps=1000, atol=1e-6, rtol=1e-6, verbose=0, infval=100)
    nμ = n_species(M)

    if preprocess
        structure_mask, element_mask, new_idxs = preprocess_optimization(M, [i])
        M = M[structure_mask, element_mask]
        i = new_idxs[1]

        missing_pars = findall(.!element_mask)
        if !isempty(missing_pars)
            nμ = nμ - sum(missing_pars .<= nμ)
        end
    end

    nstructs, npars = size(M)

    if isnothing(σs)
        σs = ones(nstructs)
    elseif preprocess
        σs = σs[structure_mask]
    end

    if npars > 1
        x = Variable(npars)
        A = M .- M[i, :]'
        A = A[1:end .!= i, :]
        s = σs[i] ./ σs[1:end .!= i]
        ns = sum(M[:, 1:nμ]; dims=2)

        # problem = minimize(Convex.logsumexp(A * x + log.(s)),
        #                    Convex.logsumexp(M * x - log.(σs) + log.(ns)) <= log(max_ϕ),
        #                    sum(x[(nμ + 1):end]) == max_ε * (npars - nμ))
        c = Convex.logsumexp(A * x + log.(s))
        phi = Convex.logsumexp(M * x - log.(σs) + log.(ns))
        problem = minimize(c,
                           phi <= log(max_ϕ),
                           sum(x[(nμ + 1):end]) == max_ε * (npars - nμ))
        Convex.solve!(problem,
               Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "verbose" => verbose,
                                                  "eps_abs" => atol, "eps_rel" => rtol,
                                                  "max_iters" => max_steps))
        xi = vec(x.value)
        residual = problem.optval
    else
        xi = [0.0]
        residual = -Inf
    end

    if preprocess
        for mi in missing_pars
            insert!(xi, mi, -Inf)
        end
    end

    return infapprox(xi, infval), residual
end

function convex_multidesign(M, idxs; max_ε=1, max_ϕ=1, σs=nothing, preprocess=true, max_steps=1000, atol=1e-6, rtol=1e-6, verbose=0, infval=100)
    nμ = n_species(M)

    if preprocess
        structure_mask, element_mask, new_idxs = preprocess_optimization(M, idxs)
        M = M[structure_mask, element_mask]
        idxs = new_idxs
        i = first(idxs)

        missing_pars = findall(.!element_mask)
        if !isempty(missing_pars)
            nμ = nμ - sum(missing_pars .<= nμ)
        end
    end

    nstructs, npars = size(M)

    if isnothing(σs)
        σs = ones(nstructs)
    elseif preprocess
        σs = σs[structure_mask]
    end

    if npars > 1
        x = Variable(npars)
        t = Variable(1)
        A = M .- M[i, :]'
        A = A[1:end .!= i, :]
        s = σs[i] ./ σs[1:end .!= i]
        ns = sum(M[:, 1:nμ]; dims=2)

        # problem = minimize(Convex.logsumexp(A * x + log.(s)),
        #                    Convex.logsumexp(M * x - log.(σs) + log.(ns)) <= log(max_ϕ),
        #                    sum(x[(nμ + 1):end]) == max_ε * (npars - nμ))
        c = Convex.logsumexp(A * x + log.(s))
        phi = Convex.logsumexp(M * x - log.(σs) + log.(ns))
        problem = minimize(c,
                           phi <= log(max_ϕ),
                           sum(x[(nμ + 1):end]) == max_ε * (npars - nμ),
                           M[idxs, :] * x < t,
                           M[idxs, :] * x > t) 
        Convex.solve!(problem,
               Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "verbose" => verbose,
                                                  "eps_abs" => atol, "eps_rel" => rtol,
                                                  "max_iters" => max_steps))
        xi = vec(x.value)
        residual = problem.optval
    else
        xi = [0.0]
        residual = -Inf
    end

    if preprocess
        for mi in missing_pars
            insert!(xi, mi, -Inf)
        end
    end

    return infapprox(xi, infval), residual
end