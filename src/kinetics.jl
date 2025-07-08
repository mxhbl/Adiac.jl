
function cleave(anatomy::AbstractGraph, es::Vector{<:AbstractEdge})
    for edge in es
        revedge = reverse(edge)
        if revedge ∉ edges(anatomy)
            error("Cannot cleave a nondirected (interior) edge.")
        end

        anatomy = copy(anatomy)

        rem_edge!(anatomy, edge)
        rem_edge!(anatomy, revedge)
    end

    gs = NautyDiGraph[]
    comps = connected_components(anatomy)
    for comp in comps
        g = anatomy[comp]
        @views g.labels = anatomy.labels[comp]
        push!(gs, g)
    end    
    return anatomy, gs, comps
end
cleave(anatomy::AbstractGraph, edge::AbstractEdge) = cleave(anatomy, [edge])


function exterior_edges(anatomy)
    return filter!(e->(e.src >= e.dst) && reverse(e)∈edges(anatomy), edges(anatomy))
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

function generate_reactionnetwork(strs, assembly_system; maxlevel)
    gs = [s.anatomy for s in strs]
    ids = Dict(ghash(g)=>i for (i, g) in enumerate(gs))

    M = compositions(strs, assembly_system)
    B = M[:, size(assembly_system)[1]+1:end]

    reactions = NTuple{3,Int}[] # reactions in the form i, j <--> k
    bonds = Vector{Int}[]
    symmetry_factors = Float64[]

    for g in gs
        graph_reacts = generate_reactions(g; maxlevel)
        isempty(graph_reacts) && continue
        
        for greact in graph_reacts
            react = map(x->ids[ghash(x)], greact)
            b = listbonds(B[react[3], :] - B[react[1], :] - B[react[2], :])
            symfac = inv(strs[react[3]].σ)

            push!(reactions, react)
            push!(bonds, b)
            push!(symmetry_factors, symfac)
        end
    end
    return reactions, bonds, symmetry_factors
end

function generate_reactions(g; maxlevel)
    reactions = NTuple{3,Int}[]     # reactions in the form i, j <--> k
    _, halfs = generate_cuts(g; maxlevel)
    # Reaction stores graph, graph -> graph at this point
    reactions = [(g1, g2, g) for (g1, g2) in halfs]
    return reactions
end


function are_separated(vi, vj, vs1, vs2)
    return (vi ∈ vs1 && vj ∈ vs2) || 
           (vi ∈ vs2 && vj ∈ vs1)
end

function generate_cuts(g; maxlevel)
    edges = exterior_edges(g)
    cuts = Vector{eltype(edges)}[]
    halfs = Vector{typeof(g)}[]

    isempty(edges) && return cuts, halfs

    current_cut = [first(edges)]

    # Perform a depth-first backtracking search to 
    # generate all possible cuts of length <= maxlevel
    while true
        current_edge = current_cut[end]

        # Remove all edges of the current cut from the input graph and return connected components
        _, components, component_vertices = cleave(g, current_cut)

        if length(components) > 1
            # Cut was successful in separating the graph

            # Check if all vertices of the cut are in different components
            # If that is the case, add the cut to the output
            srcs = [e.src for e in current_cut]
            dsts = [e.dst for e in current_cut]
            if all(x -> are_separated(x..., component_vertices...), zip(srcs, dsts))
                push!(cuts, copy(current_cut))
                push!(halfs, components)
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

function make_kernels(reactions, agg_kernel=nothing, brk_kernel=nothing)
    if isnothing(agg_kernel)
        agg_kernel = (i, j) -> 1
    end
    if isnothing(brk_kernel)
        brk_kernel = k -> 1
    end

    ks = [agg_kernel(i, j) for (i, j, _) in reactions]
    fs = [brk_kernel(k) for (_, _, k) in reactions]
    return ks, fs
end

function kinetic_network(strs, assembly_system; maxbonds, agg_kernel=nothing, brk_kernel=nothing)
    reactions, bonds, symfacs = generate_reactionnetwork(strs, assembly_system; maxlevel=maxbonds)

    ks, fs = make_kernels(reactions, agg_kernel, brk_kernel)

    nonzero_rs = filter(r->ks[r] ≉ 0 || fs[r] ≉ 0, eachindex(reactions))
    reactions = reactions[nonzero_rs]
    bonds = bonds[nonzero_rs]
    symfacs = symfacs[nonzero_rs]

    rotation_factor = 1 #Roly.dimension(eltype(strs)) == 2 ? 2π : 8π^2

    function update_step!(du, u, p, t)
        α, εs = p[1], @view p[3:end]
        du .= 0

        for r in eachindex(reactions)
            i, j, k = reactions[r]
            bs = bonds[r]
            sym = symfacs[r]

            Ka = α * ks[r] * u[i] * u[j] * sym
            Kb = rotation_factor * exp(sum(-εs[b] for b in bs)) * fs[r] * u[k]

            Rij = Kb - Ka # We don't need a factor of two for i == j, because we add it twice in that case!
            Rk = Ka - Kb

            du[i] += Rij
            du[j] += Rij
            du[k] += Rk
        end
        return
    end
    return update_step!
end

function stochastic_network(strs, assembly_system; agg_kernel=nothing, brk_kernel=nothing, maxbonds)
    reactions, bonds, symfacs = generate_reactionnetwork(strs, assembly_system; maxlevel=maxbonds)

    ks, fs = make_kernels(reactions, agg_kernel, brk_kernel)

    nonzero_rs = filter(r->ks[r] ≉ 0 || fs[r] ≉ 0, eachindex(reactions))
    reactions = reactions[nonzero_rs]
    bonds = bonds[nonzero_rs]
    symfacs = symfacs[nonzero_rs]

    function reaction_weight(r, dir, u, p)
        i, j, k = reactions[r]
        bs = bonds[r]
        sym = symfacs[r]
        α, V, εs = p[1], p[2], @view p[3:end]

        δ = exp(sum(-εs[b] for b in bs))

        if dir == 1
            pref = i != j ? 1.0 : (u[i] > 1 ? 1.0 : 0.0)
            return α / V * pref * ks[r] * u[i] * u[j] * sym
        elseif dir == 2
            return δ * fs[r] * u[k]
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

function kinetic_simulate(strs, sys, u0, Ts, p; agg_kernel=nothing, brk_kernel=nothing, maxbonds=Inf, ctime=(Ts[2] - Ts[1])/1000)
    step = kinetic_network(strs, sys; agg_kernel, brk_kernel, maxbonds)

    prob = ODEProblem(step, u0, Ts, p)
    sol = solve(prob, Rodas5(), saveat=[0; 1e-3; ctime:ctime:Ts[2]])

    ts = sol.t
    us = reduce(hcat, sol.u)
    return us, ts
end

function stochastic_simulate(strs, assembly_system, u0, Ts, p; agg_kernel=nothing, brk_kernel=nothing, maxbonds=Inf, rng=Random.default_rng(), nsteps=100_000, cinterval=max(nsteps÷100, 1))
    step = stochastic_network(strs, assembly_system; agg_kernel, brk_kernel, maxbonds)

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
