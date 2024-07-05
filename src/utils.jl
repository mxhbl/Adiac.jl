normal_vec(x::SVector{2,F}) where F = SVector{2,F}(-x[2], x[1])

function rotate(x, ϕ)
    c, s = cospi(ϕ), sinpi(ϕ)
    #TODO performance!
    return typeof(x)([c * x[1] - s * x[2], s * x[1] + c * x[2]])
end

function flatten_coords(xs, ψs)
    # TODO generalize to 3D
    ξs = zeros(eltype(xs[1]), 2*length(xs) + length(ψs))
    for i in eachindex(xs)
        j = 1 + (i-1)*3
        @views ξs[j:j+1] .= xs[i]
        ξs[j+2] = ψs[i].θ
    end
    return ξs
end

function infapprox(x, inf_val=99.9)
    return replace(x, Inf => inf_val, -Inf => -inf_val)
end


sigmas(ps::AbstractVector{<:Polyform}) = Roly.symmetry_number.(ps)

function composition(p::Polyform, assembly_system::AssemblySystem)
    n, k = size(assembly_system)
    m = zeros(Int, n + k)

    spcs = Roly.species(p)
    for s in spcs
        m[s] += 1
    end
    es = Graphs.edges(p.anatomy)
    double_bonds = [e for e in es if reverse(e) in es]
    bonds = []
    for b in double_bonds
        if b ∉ bonds && reverse(b) ∉ bonds
            push!(bonds, b)
        end
    end

    # TODO: optimize and simplify
    intmat_idxs = findall(Roly.intmat(assembly_system))
    filter!(x->x[1] <= x[2], intmat_idxs)
    bond_idxs = []
    for b in bonds
        spcs1 = Roly.species(p)[p.encoder.bwd[b.src][1]]
        site1 = p.encoder.bwd[b.src][2]
        i = Roly.spcs_site_to_siteidx(spcs1, site1, assembly_system)

        spcs2 = Roly.species(p)[p.encoder.bwd[b.dst][1]]
        site2 = p.encoder.bwd[b.dst][2]
        j = Roly.spcs_site_to_siteidx(spcs2, site2, assembly_system)

        if i > j
            k = i
            i = j
            j = k
        end

        push!(bond_idxs, findfirst(x-> x==CartesianIndex(i, j), intmat_idxs))
    end
    for bi in bond_idxs
        if isnothing(bi)
            continue
        end
        m[n + bi] += 1
    end
    return m
end
compositions(ps::AbstractVector{<:Polyform}, sys::AssemblySystem) = reduce(vcat, composition.(ps, Ref(sys))')

