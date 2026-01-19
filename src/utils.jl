function n_species(M::AbstractMatrix)
    rows = sort(eachrow(M), by=sum)
    nμ = 0
    while sum(rows[nμ+1]) == 1
        nμ += 1
    end
    return nμ
end

function exterior_edges(anatomy)
    return filter!(e->(e.src >= e.dst) && reverse(e)∈edges(anatomy), collect(edges(anatomy)))
end

normal_vec(x::SVector{2,F}) where F = SVector{2,F}(-x[2], x[1])

function rotate(x, ϕ)
    s, c = sincospi(ϕ)
    return [c * x[1] - s * x[2], s * x[1] + c * x[2]]
end

function combinecoords(xs::AbstractVector{<:AbstractVector{F}}, ψs) where {F}
    n = length(xs)
    d = length(first(xs))
    dr = first(ψs) isa Roly.Angle ? 1 : 4

    ξs = zeros(F, d + dr, n)
    for i in axes(ξs, 2)
        ξs[1:d, i] .= xs[i]
        ξs[d+1:end, i] .= Roly.value(ψs[i])
    end
    return ξs
end

function infapprox(x, inf_val=99.9)
    return replace(x, Inf => inf_val, -Inf => -inf_val)
end