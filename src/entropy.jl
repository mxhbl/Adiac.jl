
function map_potentials(bond_potential::Function, p::Polyform, geometries::AbstractVector{<:PolygonGeometry}; energy_kwargs...)
    # TODO: Optimize
    A = adjacency_matrix(p.anatomy)
    a = A .* A'
    bond_vertices = filter(ci->ci[1]<ci[2], findall(a .> 0))
    bonds = Tuple{Int,Int,Int,Int}[]
    for ci in bond_vertices
        i, si = p.encoder.bwd[ci[1]][1:2]
        j, sj = p.encoder.bwd[ci[2]][1:2]
        push!(bonds, (i, j, si, sj))
    end

    function energy_fn(ξs::AbstractVector{<:Real})
        e = 0
        for (i, j, si, sj) in bonds
            x = 1 + (i-1) * 3
            y = 1 + (j-1) * 3
            e += bond_potential(SVector(ξs[x], ξs[x+1]), SVector(ξs[y], ξs[y+1]), ξs[x+2], ξs[y+2], geometries[i], geometries[j], si, sj; energy_kwargs...)
        end
        return e
    end

    return energy_fn
end
map_potentials(bond_potential, p, geometry::PolygonGeometry; kwargs...) = map_potentials(bond_potential, p, fill(geometry, size(p)); kwargs...)

function polyform_hessian(bond_potential::Function, p::Polyform, geometries::AbstractVector{<:PolygonGeometry}; regularizer::Real=1, energy_kwargs...)
    energy_fn = map_potentials(bond_potential, p, geometries; energy_kwargs...)
    xs0 = flatten_coords(p.xs, p.ψs)
    H = ForwardDiff.hessian(x -> energy_fn(x) / regularizer, xs0)
    return H
end
polyform_hessian(bond_potential, p, geometries::PolygonGeometry; kwargs...) = polyform_hessian(bond_potential, p, fill(geometry, size(p)); kwargs...)

function eigencoords_to_relcoords(vs::AbstractMatrix, ξs::AbstractVector, xs0::AbstractVector)
    x_com = ξs[1:2]
    ψ_com = ξs[3]
    ws = ξs[4:end] # Vibrational Coords

    n = length(xs0)
    xs = xs0 + vs[:, 4:end] * ws
    xs_trans = reduce(vcat, [(rotate(SVector(xs[i], xs[i+1]), ψ_com) + x_com)..., xs[i+2] + ψ_com] for i in 1:3:n) # TODO: optimize
    return xs_trans
end

function entropy(H::AbstractMatrix, xs::AbstractArray{<:Real}; atol=1e-6, regularizer::Real=1, β::Real=1, energy_kwargs...)
    λs, vs = eigen(H)
    @assert all(abs.(λs[1:3]) / sum(H) .< atol)
    λs *= regularizer

    S_vib = 0.5 * sum(log.(2π./β./λs[4:end]))

    ctransform(ξs) = eigencoords_to_relcoords(vs, ξs, xs)
    jac(ϕ) = abs(det(ForwardDiff.jacobian(ctransform, [0.; 0.; ϕ; zeros(length(xs) - 3)])))

    Z_rot, err = quadgk(jac, 0, 2, atol=atol)
    @assert err < atol

    # CAREFUL ABOUT DISTINGUISHING SYMMETRY NUMBER 
    σ = size(p) > 1 ? p.σ : 1
    S_rot = log(π * Z_rot / σ)
    # S_rot = log(π * Z_rot)
    
    return S_vib, S_rot
end