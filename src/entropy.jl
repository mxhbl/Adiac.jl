
function map_potentials(bond_potential::Function, p::Polyform, sys::AssemblySystem; energy_kwargs...)
    n = size(p)
    es = exterior_edges(p.anatomy)
    bonds = ((Roly.vertex2particle(p, sys, e.src), Roly.vertex2particle(p, sys, e.dst)) for e in es)
    geoms = sys.geometries
    spcs = Roly.species(p)

    d = Roly.dimension(p)
    function energy_fn(ξs::AbstractMatrix{<:Real})
        size(ξs) == (3, n) || throw(ArgumentError("Invalid coordinates"))
        E = 0
        for ((i, si), (j, sj)) in bonds
            xi, ψi = @views ξs[1:d, i], ξs[d+1:end, i]
            xj, ψj = @views ξs[1:d, j], ξs[d+1:end, j]

            E += bond_potential(xi, xj, ψi, ψj, geoms[spcs[i]], geoms[spcs[j]], si, sj; energy_kwargs...)
        end
        return E
    end

    return energy_fn
end

function polyform_hessian(bond_potential::Function, p::Polyform, sys::AssemblySystem;  energy_kwargs...)
    energy_fn = map_potentials(bond_potential, p, sys; energy_kwargs...)
    ξ0 = combinecoords(p.xs, p.ψs)
    H = ForwardDiff.hessian(x -> energy_fn(x), ξ0)
    return H
end

function comcoords2abscoords(V, ξcom, ξ0)
    dt, n = size(ξ0)
    if dt == 3
        d, dr = 2, 1
    elseif dt == 7
        d, dr = 3, 4
    else
        error()
    end

    xcom = ξcom[1:d]
    ψcom = ξcom[d+1:d+dr]
    ws = ξcom[d+dr+1:end] # Vibrational Coords

    ξabs = @views ξ0 + reshape(V[:, d+dr+1:end] * ws, dt, n)
    
    for i in axes(ξabs, 2)
        ξabs[1:d, i] .= rotate(ξabs[1:d, i], only(ψcom)) + xcom # TODO: assumes 2d
        @views ξabs[d+1:end, i] .+= ψcom
    end
    return ξabs
end

function entropy(p::Polyform{D}, sys::AssemblySystem; potential=twospring_bond, atol=1e-6, potential_kwargs...) where {D}
    n = size(p)
    dr = D == 2 ? 1 : 4
    H = polyform_hessian(potential, p, sys; potential_kwargs...)
    ξ0 = combinecoords(p.xs, p.ψs)

    λs, vs = eigen(H)
    # @show λs
    @assert all(abs.(λs[1:D+dr]) .< atol)

    S_vib = -0.5 * sum(log, λs[D+dr+1:end] / (2π); init=0)

    Otrans = zeros(D)
    Ovib = zeros(length(λs) - (D+dr))
    ctransform(ξs) = comcoords2abscoords(vs, ξs, ξ0)
    jac2d(ψ, p) = abs(det(ForwardDiff.jacobian(ctransform, [Otrans; ψ; Ovib])))

    function jac3d(θ, p)
        α, β, γ = θ
        sa, ca = sincos(α)
        sb, cb = sincos(β)
        sc, cc = sincos(γ)

        ψ = [ca, sa*cb, sa*sb*cc, sa*sb*sc]
        return abs(det(ForwardDiff.jacobian(ctransform, [Otrans; ψ; Ovib]))) * sa^2 * sb
    end
    # Z_rot, _ = quadgk(jac, 0, 2, atol=atol)

    if D == 2
        bounds = (0, 2)
        prob = IntegralProblem(jac2d, bounds)
        Z_rot = π^n * solve(prob, QuadGKJL(); abstol=atol).u
    else
        bounds = (zeros(3), [π/2, π, 2π])
        prob = IntegralProblem(θ->jac3d, bounds)
        Z_rot = solve(prob, HCubatureJL(); abstol=atol).u
    end

    # # CAREFUL ABOUT DISTINGUISHING SYMMETRY NUMBER 
    # σ = size(p) > 1 ? p.σ : 1
    σ = p.σ
    S_rot = log(Z_rot / σ)
    
    return S_vib, S_rot
end

function chiralcopy(g::G) where {G<:AbstractGraph}
    n = nv(g)
    h = G(n)
    for e in edges(g)
        add_edge!(h, reverse(e))
    end
    return h
end

function symmetrynumber3D(p::Polyform{2})
    a = p.anatomy
    b = chiralcopy(a)
    n = nv(a)

    A = blockdiag(a, b)
    for i in vertices(a)
        add_edge!(A, i, i+n)
        add_edge!(A, i+n, i)
    end

    _, autg = nauty(A)
    return convert(Int, autg.n)
end