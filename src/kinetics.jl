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


function generate_reactionnetwork(strs; maxlevel, aggkernel=nothing, brkkernel=aggkernel)
    gs = [s.anatomy for s in strs]
    ids = Dict(ghash(g)=>i for (i, g) in enumerate(gs))

    reactions = NTuple{3,Int}[] # reactions in the form i, j <--> k
    cuts = []
    components = []

    for g in gs
        _cuts, _components = generate_cuts(g; maxlevel)
        graph_reacts = [(g[c1], g[c2], g) for (c1, c2) in _components]

        isempty(graph_reacts) && continue
        
        for (cut, component, greact) in zip(_cuts, _components, graph_reacts)
            react = map(x->ids[ghash(x)], greact)

            push!(reactions, react)
            push!(cuts, cut)
            push!(components, component)
        end
    end

    ks, fs = make_kernels(reactions, cuts, components, aggkernel, brkkernel)

    nonzero_rs = filter(r->ks[r] ≉ 0 || fs[r] ≉ 0, eachindex(reactions))
    reactions = reactions[nonzero_rs]
    ks = ks[nonzero_rs]
    fs = fs[nonzero_rs]

    return reactions, ks, fs
end

function kinetic_network(assembly_system, ξ, Zs; maxbonds, kernel, brkkernel=kernel)
    strs = polygen(assembly_system)
    M = compositions(strs, assembly_system)

    log_ρs = logdensities(ξ, M, Zs)

    reactions, ks, fs = generate_reactionnetwork(strs; maxlevel=maxbonds, aggkernel=kernel, brkkernel)
    fs = [fs[r] * exp(log_ρs[i] + log_ρs[j] - log_ρs[k]) for (r, (i,j,k)) in enumerate(reactions)]
    kscale = StatsBase.geomean(fs)
    fs /= kscale

    function update_step!(du, u, p, t)
        du .= 0

        for r in eachindex(reactions)
            i, j, k = reactions[r]

            Ka = ks[r] * u[i] * u[j]
            Kb = fs[r] * u[k]

            Rij = Kb - Ka # We don't need a factor of 2 for i == j, because we add it twice in that case!
            Rk = Ka - Kb

            du[i] += Rij
            du[j] += Rij
            du[k] += Rk
        end
        return
    end
    return update_step!, kscale
end

# function stochastic_network(strs, assembly_system; aggkernel=nothing, brkkernel=nothing, maxbonds)
#     reactions, bonds, symfacs, ks, fs = generate_reactionnetwork(strs, assembly_system; maxlevel=maxbonds, aggkernel, brkkernel)

#     function reaction_weight(r, dir, u, p)
#         i, j, k = reactions[r]
#         bs = bonds[r]
#         sym = symfacs[r]
#         V, εs = p[1], @view p[2:end]

#         if dir == 1
#             pref = i != j ? 1.0 : (u[i] > 1 ? 1.0 : 0.0)
#             return inv(V) * pref * ks[r] * u[i] * u[j]
#         elseif dir == 2
#             return 8π^2 * exp(sum(-εs[b] for b in bs)) * fs[r] * u[k] / sym
#         end
#         error()
#         return 
#     end

#     ws = zeros(length(reactions), 2)

#     function update_step!(rng, u, p, t)    
#         for r in eachindex(reactions), d in (1, 2)
#             ws[r, d] = reaction_weight(r, d, u, p)
#         end

#         wsum = sum(ws)
#         τ = inv(wsum) * log(inv(rand(rng)))

#         ci = sample(rng, vec(CartesianIndices(ws)), Weights(vec(ws), wsum))
#         r, dir = ci[1], ci[2]
#         i, j, k = reactions[r]
#         if dir == 1 
#             u[i] -= 1
#             u[j] -= 1
#             u[k] += 1
#         elseif dir == 2
#             u[i] += 1
#             u[j] += 1
#             u[k] -= 1
#         else
#             error()
#         end
#         return τ
#     end
#     return update_step!
# end

function kinetic_simulate(sys, ξ; Zs, Ts, kernel, brkkernel=kernel, maxbonds=Inf, saveat=[])
    np = size(sys)[1]
    M = compositions(polygen(sys), sys)
    nstr = size(M, 1)

    step, kscale = kinetic_network(sys, ξ, Zs; kernel, brkkernel, maxbonds)
    tscale = inv(kscale)
    ρscale = kscale

    ρ0 = vcat(monomer_densities(ξ, M, Zs), zeros(nstr - np)) / ρscale
    Ts = Ts ./ tscale
    saveat = saveat ./ tscale

    prob = ODEProblem(step, ρ0, Ts)
    sol = solve(prob, Rodas5P(); saveat=saveat)

    ts = sol.t * tscale
    us = reduce(hcat, sol.u * ρscale)
    return us, ts
end

# function stochastic_simulate(strs, assembly_system, u0, Ts, p; aggkernel=nothing, brkkernel=nothing, maxbonds=Inf, rng=Random.default_rng(), nsteps=100_000, cinterval=max(nsteps÷100, 1))
#     step = stochastic_network(strs, assembly_system; aggkernel, brkkernel, maxbonds)

#     ts = zeros(nsteps ÷ cinterval)
#     us = zeros(Int, length(u0), nsteps ÷ cinterval)

#     u = copy(u0)
#     us[:, 1] .= u

#     t = ts[1] = Ts[1]
#     for i in 2:nsteps
#         dt = step(rng, u, p, t)
#         t += dt
#         if i % cinterval == 0
#             us[:, i÷cinterval] .= u
#             ts[i÷cinterval] = t
#         end
#         if t >= Ts[2]
#             ts = ts[1:i÷cinterval]
#             us = us[:, 1:i÷cinterval]
#             break
#         end
#     end
#     return us, ts
# end

function stability_matrix(assembly_system; symmetrize=false, kernel, brkkernel=kernel, Zs, maxbonds)
    strs = polygen(assembly_system)

    reactions, ks, fs = generate_reactionnetwork(strs; maxlevel=maxbonds, aggkernel=kernel, brkkernel)
    M = compositions(strs, assembly_system)

    function Sfn!(S, ξ)
        S .= 0
        ρeq = densities(ξ, M, Zs)

        for r in eachindex(reactions)
            i, j, k = reactions[r]

            Ka = ks[r]
            Kb = (ρeq[i] * ρeq[j] / ρeq[k]) * fs[r]

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
        return S 
    end

    function ∂S∂μfn!(S, ξ)
        S .= 0
        ρeq = densities(ξ, M, Zs)
        ∂ρeq = permutedims(∂ρ∂μ(ξ, M, Zs))

        scratch = zero(ξ)

        for r in eachindex(reactions)
            i, j, k = reactions[r]

            Ka = ks[r]
            Kb = fs[r] 

            @views @. begin 
                scratch = -Ka * ∂ρeq[:, j] 
                S[i, i, :] += scratch
                S[j, i, :] += scratch
                S[k, i, :] -= scratch


                scratch = -Ka * ∂ρeq[:, i] 
                S[i, j, :] += scratch
                S[j, j, :] += scratch
                S[k, j, :] -= scratch

                scratch = Kb *(-ρeq[i] * ρeq[j] / ρeq[k]^2 * ∂ρeq[:, k] + 
                            ∂ρeq[:, i] * ρeq[j] / ρeq[k] 
                            + ∂ρeq[:, j] * ρeq[i] / ρeq[k])
                S[i, k, :] += scratch
                S[j, k, :] += scratch
                S[k, k, :] -= scratch
            end
        end
        return S
    end

    #####################
    #### Return inv(D) S D, where D = sqrt(ρᵢ)
    function Ssym_fn!(S, ξ; scale=1)
        ρeq = densities(ξ, M, Zs)
        ρeq_sqrt = sqrt.(ρeq)

        S .= 0
        for r in eachindex(reactions)
            i, j, k = reactions[r]

            Ka = ks[r] * scale
            Kb = fs[r] * scale

            ij = ρeq_sqrt[i] * ρeq_sqrt[j] 
            ik = ρeq_sqrt[i] / ρeq_sqrt[k] * ρeq[j]
            jk = ρeq_sqrt[j] / ρeq_sqrt[k] * ρeq[i]

            S[i, i] += -Ka * ρeq[j]
            S[i, j] += -Ka * ij
            S[i, k] += Kb * ik

            S[j, i] += -Ka * ij
            S[j, j] += -Ka * ρeq[i]
            S[j, k] += Kb * jk

            S[k, i] += Ka * ik
            S[k, j] += Ka * jk
            S[k, k] += -Kb * (ρeq[i] * ρeq[j] / ρeq[k])
        end
        return S
    end

    function ∂Ssym∂μ_fn!(S, ξ; scale=1)
        ρeq = densities(ξ, M, Zs)
        ρeq_sqrt = sqrt.(ρeq)
        ∂ρeq = permutedims(∂ρ∂μ(ξ, M, Zs))

        S .= 0

        scratch = zero(ξ)

        for r in eachindex(reactions)
            i, j, k = reactions[r]

            Ka = ks[r] * scale
            Kb = fs[r] * scale

            @views @. begin
                scratch = -Ka * ((ρeq_sqrt[j] / ρeq_sqrt[i]) * ∂ρeq[:, i] + (ρeq_sqrt[i] / ρeq_sqrt[j]) * ∂ρeq[:, j]) / 2
                S[i, j, :] += scratch
                S[j, i, :] += scratch

                # ik
                scratch = (ρeq_sqrt[i] / ρeq_sqrt[k] * ∂ρeq[:, j] + ρeq[j] / (2ρeq_sqrt[i] * ρeq_sqrt[k]) * ∂ρeq[:, i] -
                    ρeq_sqrt[i] * ρeq[j] / (2ρeq_sqrt[k]^3) * ∂ρeq[:, k])
                S[i, k, :] += Kb * scratch
                S[k, i, :] += Ka * scratch

                # jk
                scratch = (ρeq_sqrt[j] / ρeq_sqrt[k] * ∂ρeq[:, i] + ρeq[i] / (2ρeq_sqrt[j] * ρeq_sqrt[k]) * ∂ρeq[:, j] -
                ρeq_sqrt[j] * ρeq[i] / (2ρeq_sqrt[k]^3) * ∂ρeq[:, k])

                S[j, k, :] += Kb * scratch
                S[k, j, :] += Ka * scratch

                S[i, i, :] += -Ka * ∂ρeq[:, j] 
                S[j, j, :] += -Ka * ∂ρeq[:, i] 
                S[k, k, :] += -Kb * (-ρeq[i] * ρeq[j] / ρeq[k]^2 * ∂ρeq[:, k] + ∂ρeq[:, i] * ρeq[j] / ρeq[k] + ∂ρeq[:, j] * ρeq[i] / ρeq[k])
            end
        end
        return S
    end

    if symmetrize
        return Ssym_fn!, ∂Ssym∂μ_fn!
    else
        return Sfn!, ∂S∂μfn!
    end
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
