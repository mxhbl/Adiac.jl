function cleave(anatomy::AbstractGraph, es::Vector{<:AbstractEdge})
    anatomy = copy(anatomy)

    for edge in es
        revedge = reverse(edge)
        if revedge ∉ edges(anatomy)
            error("Cannot cleave a nondirected (interior) edge.")
        end

        rem_edge!(anatomy, edge)
        rem_edge!(anatomy, revedge)
    end

    comps = connected_components(anatomy) 
    return anatomy, comps
end
cleave(anatomy::AbstractGraph, edge::AbstractEdge) = cleave(anatomy, [edge])


function exterior_edges(anatomy)
    return filter!(e->(e.src >= e.dst) && reverse(e)∈edges(anatomy), collect(edges(anatomy)))
end

function listbonds(bondcounts)
    # Convert bond counts, i.e. [0, 1, 0, 0, 2]
    # into a list of bonds (with repeats), i.e. [2, 5, 5]
    bondlist = zeros(Int, sum(bondcounts))

    k = 1
    for (i, b) in pairs(bondcounts)
        if b > 0
            bondlist[k:k+b-1] .= i
            k += b
        end
    end
    return bondlist
end

function generate_reactions(g; maxlevel)
    cuts, components = generate_cuts(g; maxlevel)
    # Reaction stores graph, graph -> graph at this point
    reactions = [(g[c1], g[c2], g) for (c1, c2) in components]
    return reactions, cuts, components
end


function are_separated(vi, vj, vs1, vs2)
    return (vi ∈ vs1 && vj ∈ vs2) || 
           (vi ∈ vs2 && vj ∈ vs1)
end

function generate_cuts(g; maxlevel)
    edges = exterior_edges(g)
    cuts = Vector{eltype(edges)}[]
    halfs = Vector{Vector{Int32}}[]

    isempty(edges) && return cuts, halfs

    current_cut = [first(edges)]

    # Perform a depth-first backtracking search to 
    # generate all possible cuts of length <= maxlevel
    while true
        current_edge = current_cut[end]

        # Remove all edges of the current cut from the input graph and return connected components
        _, component_vertices = cleave(g, current_cut)

        if length(component_vertices) > 1
            # Cut was successful in separating the graph

            # Check if all vertices of the cut are in different components
            # If that is the case, add the cut to the output
            srcs = [e.src for e in current_cut]
            dsts = [e.dst for e in current_cut]
            if all(x -> are_separated(x..., component_vertices...), zip(srcs, dsts))
                push!(cuts, copy(current_cut))
                push!(halfs, component_vertices)
            end

            nextedge_idx = nothing # initiate an upward traverse
        elseif length(current_cut) == maxlevel
            nextedge_idx = nothing # initiate an upward traverse
        else
            # Cut is incomplete, keep traversing downward
            nextedge_idx = findfirst(e->e>current_edge, edges) # TODO use searchsorted
        end

        # If no additional edge can be added to the cut, traverse upward until a viable
        # branch is found
        while isnothing(nextedge_idx)
            isempty(current_cut) && @goto finished
            current_edge = pop!(current_cut)
            nextedge_idx = findfirst(e->e>current_edge, edges) # TODO use searchsorted
        end

        next_edge = edges[nextedge_idx]
        push!(current_cut, next_edge)
    end

    @label finished

    return cuts, halfs
end

function make_kernels(reactions, cuts, components, aggkernel=nothing, brkkernel=nothing)
    if isnothing(aggkernel)
        aggkernel = (k, cut, component) -> 1
    end
    if isnothing(brkkernel)
        brkkernel = (k, cut, component) -> 1
    end

    ks = [aggkernel(k, cut, component) for ((_, _, k), cut, component) in zip(reactions, cuts, components)]
    fs = [brkkernel(k, cut, component) for ((_, _, k), cut, component) in zip(reactions, cuts, components)]
    return ks, fs
end


function generate_reactionnetwork(strs, assembly_system; maxlevel, aggkernel=nothing, brkkernel=nothing)
    gs = [s.anatomy for s in strs]
    ids = Dict(ghash(g)=>i for (i, g) in enumerate(gs))

    M = compositions(strs, assembly_system)
    B = M[:, size(assembly_system)[1]+1:end]

    reactions = NTuple{3,Int}[] # reactions in the form i, j <--> k
    cuts = []
    components = []
    bonds = Vector{Int}[]
    symmetry_factors = Float64[]

    for g in gs
        _cuts, _components = generate_cuts(g; maxlevel)
        graph_reacts = [(g[c1], g[c2], g) for (c1, c2) in _components]

        isempty(graph_reacts) && continue
        
        for (cut, component, greact) in zip(_cuts, _components, graph_reacts)
            react = map(x->ids[ghash(x)], greact)
            b = listbonds(B[react[3], :] - B[react[1], :] - B[react[2], :])
            symfac = strs[react[1]].σ * strs[react[2]].σ / strs[react[3]].σ

            push!(reactions, react)
            push!(bonds, b)
            push!(symmetry_factors, symfac)
            push!(cuts, cut)
            push!(components, component)
        end
    end

    ks, fs = make_kernels(reactions, cuts, components, aggkernel, brkkernel)

    nonzero_rs = filter(r->ks[r] ≉ 0 || fs[r] ≉ 0, eachindex(reactions))
    reactions = reactions[nonzero_rs]
    bonds = bonds[nonzero_rs]
    symmetry_factors = symmetry_factors[nonzero_rs]
    ks = ks[nonzero_rs]
    fs = fs[nonzero_rs]

    return reactions, bonds, symmetry_factors, ks, fs
end

function kinetic_network(strs, assembly_system; maxbonds, aggkernel=nothing, brkkernel=nothing, k0=1)
    reactions, bonds, symfacs, ks, fs = generate_reactionnetwork(strs, assembly_system; maxlevel=maxbonds, aggkernel, brkkernel)

    function update_step!(du, u, p, t)
        εs = @view p[2:end]
        du .= 0

        for r in eachindex(reactions)
            i, j, k = reactions[r]
            bs = bonds[r]
            sym = symfacs[r]

            Ka = k0 * ks[r] * u[i] * u[j]
            # Kb = 8π^2 * exp(sum(-εs[b] for b in bs)) * fs[r] * u[k] / sym
            Kb = exp(log(k0) + log(8π^2) + sum(-εs[b] for b in bs) + log(fs[r])) * u[k] / sym

            Rij = Kb - Ka # We don't need a factor of 2 for i == j, because we add it twice in that case!
            Rk = Ka - Kb

            du[i] += Rij
            du[j] += Rij
            du[k] += Rk
        end
        return
    end
    return update_step!
end

function stochastic_network(strs, assembly_system; aggkernel=nothing, brkkernel=nothing, maxbonds)
    reactions, bonds, symfacs, ks, fs = generate_reactionnetwork(strs, assembly_system; maxlevel=maxbonds, aggkernel, brkkernel)

    function reaction_weight(r, dir, u, p)
        i, j, k = reactions[r]
        bs = bonds[r]
        sym = symfacs[r]
        V, εs = p[1], @view p[2:end]

        if dir == 1
            pref = i != j ? 1.0 : (u[i] > 1 ? 1.0 : 0.0)
            return inv(V) * pref * ks[r] * u[i] * u[j]
        elseif dir == 2
            return 8π^2 * exp(sum(-εs[b] for b in bs)) * fs[r] * u[k] / sym
        end
        error()
        return 
    end

    ws = zeros(length(reactions), 2)

    function update_step!(rng, u, p, t)    
        for r in eachindex(reactions), d in (1, 2)
            ws[r, d] = reaction_weight(r, d, u, p)
        end

        wsum = sum(ws)
        τ = inv(wsum) * log(inv(rand(rng)))

        ci = sample(rng, vec(CartesianIndices(ws)), Weights(vec(ws), wsum))
        r, dir = ci[1], ci[2]
        i, j, k = reactions[r]
        if dir == 1 
            u[i] -= 1
            u[j] -= 1
            u[k] += 1
        elseif dir == 2
            u[i] += 1
            u[j] += 1
            u[k] -= 1
        else
            error()
        end
        return τ
    end
    return update_step!
end

function kinetic_simulate(strs, sys, u0, Ts, p; aggkernel=nothing, brkkernel=nothing, maxbonds=Inf, k0=1, saveat=[])
    step = kinetic_network(strs, sys; aggkernel, brkkernel, maxbonds, k0)

    prob = ODEProblem(step, u0, Ts ./ k0, p)
    sol = solve(prob, Rodas5(); saveat=saveat/k0)

    ts = sol.t * k0
    us = reduce(hcat, sol.u)
    return us, ts
end

function stochastic_simulate(strs, assembly_system, u0, Ts, p; aggkernel=nothing, brkkernel=nothing, maxbonds=Inf, rng=Random.default_rng(), nsteps=100_000, cinterval=max(nsteps÷100, 1))
    step = stochastic_network(strs, assembly_system; aggkernel, brkkernel, maxbonds)

    ts = zeros(nsteps ÷ cinterval)
    us = zeros(Int, length(u0), nsteps ÷ cinterval)

    u = copy(u0)
    us[:, 1] .= u

    t = ts[1] = Ts[1]
    for i in 2:nsteps
        dt = step(rng, u, p, t)
        t += dt
        if i % cinterval == 0
            us[:, i÷cinterval] .= u
            ts[i÷cinterval] = t
        end
        if t >= Ts[2]
            ts = ts[1:i÷cinterval]
            us = us[:, 1:i÷cinterval]
            break
        end
    end
    return us, ts
end

function stability_matrix(strs, assembly_system; aggkernel=nothing, brkkernel=nothing, maxbonds)
    ns = length(strs)
    np, nb = size(assembly_system)

    reactions, bonds, symfacs, ks, fs = generate_reactionnetwork(strs, assembly_system; maxlevel=maxbonds, aggkernel, brkkernel)
    M = compositions(strs, assembly_system)
    Zs = 8π^2 ./ [s.σ for s in strs]

    function Sfn!(S, ξ)
        S .= 0
        ρeq = densities(ξ, M, Zs)

        for r in eachindex(reactions)
            i, j, k = reactions[r]
            bs = bonds[r]
            sym = symfacs[r]

            Ka = ks[r]
            Kb = 8π^2 * exp(sum(-ξ[b+np] for b in bs)) * fs[r] / sym

            S[i, i] += -Ka * ρeq[j] 
            S[i, j] += -Ka * ρeq[i] 
            S[i, k] += Kb

            S[j, i] += -Ka * ρeq[j] 
            S[j, j] += -Ka * ρeq[i] 
            S[j, k] += Kb

            S[k, i] += Ka * ρeq[j] 
            S[k, j] += Ka * ρeq[i] 
            S[k, k] += -Kb
        end
        return S # TODO: remove this once other functions are fixed
    end

    Sfn(ξ) = Sfn!(zeros(eltype(ξ), ns, ns), ξ)

    function ∂S∂μfn!(S, ξ)
        S .= 0
        ∂ρeq = ∂ρ∂μ(ξ, M, Zs)

        for r in eachindex(reactions)
            i, j, k = reactions[r]
            bs = bonds[r]
            sym = symfacs[r]

            #TODO fix this hack
            B = zeros(eltype(ξ), np + nb)
            for b in bs 
                B[np+b] += 1
            end

            Ka = ks[r]
            Kb = -8π^2 * B * exp(sum(-ξ[b+np] for b in bs)) * fs[r] / sym

            S[i, i, :] += -Ka * ∂ρeq[j, :] 
            S[i, j, :] += -Ka * ∂ρeq[i, :] 
            S[i, k, :] += Kb

            S[j, i, :] += -Ka * ∂ρeq[j, :] 
            S[j, j, :] += -Ka * ∂ρeq[i, :] 
            S[j, k, :] += Kb

            S[k, i, :] += Ka * ∂ρeq[j, :] 
            S[k, j, :] += Ka * ∂ρeq[i, :] 
            S[k, k, :] += -Kb
        end
        return S
    end

    ∂S∂μfn(ξ) = ∂S∂μfn!(zeros(eltype(ξ), ns, ns, length(ξ)), ξ)


    # function ∂S∂ϕfn(ϕs, εs)
    #     S = zeros(eltype(ϕs), ns, ns, np + nb)
    #     ∂ρeq = ∂ρ∂ϕ(ϕs, εs, M, Zs)

    #     for r in eachindex(reactions)
    #         i, j, k = reactions[r]
    #         bs = bonds[r]
    #         sym = symfacs[r]

    #         #TODO fix this hack
    #         B = zeros(eltype(ϕs), np + nb)
    #         for b in bs 
    #             B[np+b] += 1
    #         end

    #         Ka = α * ks[r] * sym
    #         Kb = α * -rotation_factor * B * exp(sum(-εs[b] for b in bs)) * fs[r]

    #         S[i, i, :] += -Ka * ∂ρeq[j, :] 
    #         S[i, j, :] += -Ka * ∂ρeq[i, :] 
    #         S[i, k, :] += Kb

    #         S[j, i, :] += -Ka * ∂ρeq[j, :] 
    #         S[j, j, :] += -Ka * ∂ρeq[i, :] 
    #         S[j, k, :] += Kb

    #         S[k, i, :] += Ka * ∂ρeq[j, :] 
    #         S[k, j, :] += Ka * ∂ρeq[i, :] 
    #         S[k, k, :] += -Kb
    #     end
    #     return S
    # end

    return Sfn, ∂S∂μfn, Sfn!, ∂S∂μfn!
end

function τc(S; np, thresh=1e-12)
    ns = size(S, 1)

    C = maximum(S)
    S = S / C

    # λ = maximum(real, schur(S).values * C)
    λ = partialsort!(schur(S).values * C, ns-np, by=real)
    # if imag(λ) > thresh
    #     @warn "correlation time calculation leads to complex result, which have been clipped"
    # end
    return -inv(real(λ))
end

function ∂τc(S, ∂S; np, thresh=1e-12)
    ns = size(S, 1)

    C = maximum(abs, S)
    S = S / C

    sch = schur(complex(S))
    λs = sch.values
    xs = eigvecs(sch; left=false)
    ys = eigvecs(sch; left=true)

    # idx = argmax(real.(λs))
    idx = partialsortperm(λs, ns-np, by=real)

    λ, x, y = λs[idx] * C, xs[:, idx], ys[:, idx]

    ∂λ = @views [dot(y, ∂S[:, :, i], x) / (y'x) for i in axes(∂S, 3)]
    # if max(maximum(imag(∂λ)), imag(λ)) > thresh
    #     @warn "gradient calculation leads to complex derivatives, which have been clipped"
    #     @warn "imag = $(max(maximum(imag(∂λ)), imag(λ)))"
    # end

    ∂τ = real.(∂λ) / real(λ)^2
    return ∂τ
end


# function reactionweights(ξ, strs, sys; ρs=nothing, aggkernel=nothing, brkkernel=nothing, maxbonds=Inf)
#     np, nb = size(sys)

#     reactions, bonds, symmetry_factors, ks, fs = generate_reactionnetwork(strs, sys; maxlevel=maxbonds, aggkernel, brkkernel)
#     M = compositions(strs, sys)
#     Zs = 8π^2 * inv.(s.σ for s in strs)

#     if isnothing(ρs)
#         ρs = densities(ξ, M, Zs)
#     end
    
#     function reaction_weight(r)
#         i, j, k = reactions[r]
#         bs = bonds[r]
#         sym = symmetry_factors[r]

#         δ = exp(sum(-ξ[b + np] for b in bs))

#         fwd_rate = ks[r] * ρs[i] * ρs[j] 
#         bwd_rate = δ * fs[r] * ρs[k] / sym
#         return (fwd_rate, bwd_rate)
#     end

#     ws = zeros(length(reactions), 2)

#     for r in eachindex(reactions)
#         ws[r, :] .= reaction_weight(r)
#     end

#     return reactions, ws
# end
