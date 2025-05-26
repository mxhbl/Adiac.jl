function n_species(M::AbstractMatrix)
    rows = sort(eachrow(M), by=sum)
    nμ = 0
    while sum(rows[nμ+1]) == 1
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