# TODO: make this independent of Roly types
function twospring_bond(
    xi, xj, ψi, ψj,
    geom_i::Roly.PolygonGeometry, 
    geom_j::Roly.PolygonGeometry, 
    site_i::Integer, 
    site_j::Integer;
    ε::Real,
    ω::Real,
    r::Real)

    ψi, ψj = ψi[1], ψj[1]
    
    a = 2*norm(geom_i.xs[1]) / cot(π / Roly.nsites(geom_i))
    fi = normal_vec(geom_i.xs[site_i])
    fi *= a / norm(fi)
    fj = normal_vec(geom_i.xs[site_j])
    fj *= a / norm(fj)

    zsi = xi + rotate(geom_i.xs[site_i] + fi*r / 2, ψi), xi + rotate(geom_i.xs[site_i] - fi*r / 2, ψi)
    zsj = xj + rotate(geom_j.xs[site_j] - fj*r / 2, ψj), xj + rotate(geom_j.xs[site_j] + fj*r / 2, ψj)

    return -ε + 0.5 * ω^2 * (sum(x->x^2, zsi[1] - zsj[1]) + sum(x->x^2, zsi[2] - zsj[2]))
end
