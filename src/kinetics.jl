
function cleave(anatomy::AbstractGraph, edge)
    revedge = reverse(edge)
    if revedge ∉ edges(anatomy)
        error("Cannot cleave a nondirected (interior) edge.")
    end

    anatomy = copy(anatomy)

    rem_edge!(anatomy, edge)
    rem_edge!(anatomy, revedge)

    gs = NautyDiGraph[]
    comps = connected_components(anatomy)
    for comp in comps
        g = anatomy[comp]
        @views g.labels = anatomy.labels[comp]
        push!(gs, g)
    end    
    return anatomy, gs, comps
end

function interior_edges(anatomy)
    return filter!(e->(e.src >= e.dst) && reverse(e)∈edges(anatomy), edges(anatomy))
end

function list_reactions(strs)
    gs = [s.anatomy for s in strs]
    ids = Dict(ghash(g)=>i for (i, g) in enumerate(gs))

    reactions = NTuple{3,Int}[] # reactions in the form i, j <--> k
    bondbreaks = Int[]
    symfacs = Float64[]

    for g in gs
        gid = ids[ghash(g)]
        used_edges = []

        for e in interior_edges(g)
            gcleave, parts, _ = cleave(g, e)

            if length(parts) == 1
                v1, v2 = e.src, e.dst

                for e2 in interior_edges(gcleave)
                    e2 in used_edges && continue
                    gcleave2, parts2, comps = cleave(gcleave, e2)

                    length(parts2) == 1 && continue
                    (v1 ∈ comps[1] && v2 ∈ comps[1] || 
                        v1 ∈ comps[2] && v2 ∈ comps[2]) && continue
                    
                    component_ids = sort([ids[ghash(parts2[1])], ids[ghash(parts2[2])]])
                    reaction = (component_ids..., gid)
                    push!(reactions, reaction)
                    push!(bondbreaks, 2)

                    symfac = (reaction[1] == reaction[2] ? 2 : 1) / strs[gid].σ
                    push!(symfacs, symfac)
                end
            else
                component_ids = sort([ids[ghash(parts[1])], ids[ghash(parts[2])]])
                reaction = (component_ids..., gid)
                push!(reactions, reaction)
                push!(bondbreaks, 1)

                symfac = (reaction[1] == reaction[2] ? 2 : 1) / strs[gid].σ
                push!(symfacs, symfac)
            end


            push!(used_edges, e)
        end
    end
    return reactions, bondbreaks, symfacs
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

function kinetic_network(strs; agg_kernel=nothing, brk_kernel=nothing)
    reactions, bondbreaks, symfacs = list_reactions(strs)

    ks, fs = make_kernels(reactions, agg_kernel, brk_kernel)

    nonzero_rs = filter(r->ks[r] ≉ 0 || fs[r] ≉ 0, eachindex(reactions))
    reactions = reactions[nonzero_rs]
    bondbreaks = bondbreaks[nonzero_rs]
    symfacs = symfacs[nonzero_rs]

    function update_step!(du, u, p, t)
        α, δ = p
        du .= 0

        for r in eachindex(reactions)
            i, j, k = reactions[r]
            bbs = bondbreaks[r]
            sym = symfacs[r]

            if i != j
                du[i] += (-α * ks[r] * u[i] * u[j] * sym + δ^bbs * fs[r] * u[k])
                du[j] += (-α * ks[r] * u[i] * u[j] * sym + δ^bbs * fs[r] * u[k])
                du[k] += (α * ks[r] * u[i] * u[j] * sym - δ^bbs * fs[r] * u[k])
            else
                du[i] += (-2α * ks[r] * u[i]^2 * sym + 2δ^bbs * fs[r] * u[k])
                du[k] += (α * ks[r] * u[i]^2 * sym - δ^bbs * fs[r] * u[k])
            end
        end
        return
    end
    return update_step!
end

function stochastic_network(strs; agg_kernel=nothing, brk_kernel=nothing)
    reactions, bondbreaks, symfacs = list_reactions(strs)

    ks, fs = make_kernels(reactions, agg_kernel, brk_kernel)

    nonzero_rs = filter(r->ks[r] ≉ 0 || fs[r] ≉ 0, eachindex(reactions))
    reactions = reactions[nonzero_rs]
    bondbreaks = bondbreaks[nonzero_rs]
    symfacs = symfacs[nonzero_rs]

    function reaction_weight(r, dir, u, p)
        i, j, k = reactions[r]
        bbs = bondbreaks[r]
        sym = symfacs[r]
        α, δ, V = p

        if dir == 1
            pref = i != j ? 1.0 : (u[i] > 1 ? 0.5 : 0.0)
            return α / V * pref * ks[r] * u[i] * u[j] * sym
        elseif dir == 2
            return δ^bbs * fs[r] * u[k]
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

function kinetic_simulate(strs, u0, Ts, p; agg_kernel=nothing, brk_kernel=nothing, ctime=(Ts[2] - Ts[1])/1000)
    step = kinetic_network(strs; agg_kernel, brk_kernel)

    prob = ODEProblem(step, u0, Ts, p)
    sol = solve(prob, Rodas5(), saveat=[0; 1e-3; ctime:ctime:Ts[2]])

    ts = sol.t
    us = reduce(hcat, sol.u)
    return us, ts
end

function stochastic_simulate(strs, u0, Ts, p; agg_kernel=nothing, brk_kernel=nothing, rng=Random.default_rng(), nsteps=100_000, cinterval=max(nsteps÷100, 1))
    step = stochastic_network(strs; agg_kernel, brk_kernel)

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
