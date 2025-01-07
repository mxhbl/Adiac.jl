function n_species(M::AbstractMatrix)
    nμ = 0
    while sum(M[nμ + 1, :]) == 1
        nμ += 1
    end
    return nμ
end

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

    bondlist = findall(Roly.intmat(assembly_system))
    filter!(x->x[1] <= x[2], bondlist)
    sort!(bondlist)
    
    for b in bonds
        lsrc, ldst = sort([p.anatomy.labels[b.src], p.anatomy.labels[b.dst]])
        i = findfirst(x->x==CartesianIndex(lsrc, ldst), bondlist)
        m[n + i] += 1
    end

    return m
end
compositions(ps::AbstractVector{<:Polyform}, sys::AssemblySystem) = reduce(vcat, composition.(ps, Ref(sys))')

