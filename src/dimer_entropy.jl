using Roly
using ForwardDiff
using LinearAlgebra
using Adiac
using SpecialFunctions, Integrals, MCIntegration
using StaticArrays
using CairoMakie

struct Orientation{D,F<:Real,S}
    ψ::S
    function Orientation{D,F}(ψ) where {D,F}
        d = D * (D - 1) ÷ 2
        return new{D,F,SVector{d,F}}(ψ)
    end
end
Angle{F} = Orientation{2,F}
Euler{F} = Orientation{3,F}

Angle{F}(ψ::Real) where {F} = Angle{F}(SVector{1,F}(ψ))
Angle(ψ::AbstractVector) = Angle{eltype(ψ)}(only(ψ))
Angle(ψ::F) where {F} = Angle{F}(ψ)

Euler{F}(ψ₁::Real, ψ₂::Real, ψ₃::Real) where {F} = Euler{F}(SVector{3,F}(ψ₁, ψ₂, ψ₃))
Euler(ψ::AbstractVector) = Euler{eltype(ψ)}(ψ[1], ψ[2], ψ[3])
Euler(ψ₁::F1, ψ₂::F2, ψ₃::F3) where {F1,F2,F3} = Euler{promote_type(F1,F2,F3)}(ψ₁, ψ₂, ψ₃)

Base.getindex(ψ::Angle, inds...) = ψ.ψ[1]
Base.getindex(ψ::Euler, inds...) = Base.getindex(ψ.ψ, inds...)
Base.iterate(ψ::Euler, state) = Base.iterate(ψ.ψ, state)
Base.iterate(ψ::Euler) = Base.iterate(ψ.ψ)

function R(ψ::Angle)
    s, c = sincos(ψ[])
    return @SMatrix [c  -s;
                     s c]
end

function R(ψ::Euler)
    α, β, γ = ψ
    sa, ca = sincos(α)
    sb, cb = sincos(β)
    sc, cc = sincos(γ)

    return @SMatrix [ca*cb*cc - sa*sc    -cc*sa - ca*cb*sc   ca*sb;
                     ca*sc + cb*cc*sa    ca*cc - cb*sa*sc    sa*sb;
                     -cc*sb              sb*sc               cb]
end

R(x::AbstractArray) = R(Euler(x))
R(x::Number) = R(Angle(x))

function Euler(R::AbstractMatrix)
    size(R) == (3, 3) && (R' * R ≈ I) || throw(ArgumentError("R must be an orthogonal 3x3 matrix"))
    β = acos(R[3, 3])
    sb = sin(β)

    γ = atan(R[3, 2] / sb, -R[3, 1] / sb)
    α = atan(R[2, 3] / sb, R[1, 3] / sb)
    return Euler(α, β, γ)
end

function Base.inv(ψ::Euler)
    # Equivalent to Euler(R(ψ)')
    α, β, γ = ψ
    return Euler(-γ, -β, -α)
end

function Base.:(*)(ψ1::Euler, ψ2::Euler)
    return Euler(R(ψ1) * R(ψ2))
end

function Base.:(*)(ψ1::Angle, ψ2::Angle)
    return Angle((ψ1[] + ψ2[]) % 2π)
end
function Base.inv(ψ::Angle)
    return Angle(2π - ψ[])
end

function make_dimerenergy(As, Bs; k)#, offset=true)
    # the offset is needed for euler angles to avoid having equilibrium at gimbal lock
    # right now, this breaks 2d
    D = size(As, 1)

    # R_offset = !offset || D == 2 ? I : R([0, π/2, 0])
    R_offset = I

    reshape2d(x; dropfirst=false) = !dropfirst ? (x[1:2], Angle(x[3]), x[4:5], Angle(x[6])) : (zeros(2), Angle(x[1]), x[2:3], Angle(x[4]))
    reshape3d(x; dropfirst=false) = !dropfirst ? (x[1:3], Euler(x[4:6]), x[7:9], Euler(x[10:12])) : (zeros(3), Euler(x[1:3]), x[4:6], Euler(x[7:9]))

    function energy(x1, ψ1, x2, ψ2)
        aᵢ = [x1 .+ R_offset * R(ψ1) * a for a in eachcol(As)]
        bᵢ = [x2 .+ R_offset * R(ψ2) * b for b in eachcol(Bs)]
        # @show aᵢ
        # @show bᵢ
        return 0.5 * k * sum(norm((a - b))^2 for (a,b) in zip(aᵢ, bᵢ))
    end

    return energy, D == 2 ? reshape2d : reshape3d
end
function entropy2d(A, k)
    n = size(A, 2)
    K = k*n

    abar = k * sum(A, dims=2) / K
    C = k * (A .- abar) * (A .- abar)'

    t = tr(C)
    return (2π)^3/K * exp(-t) * besseli(0, t)
end
function entropy2d_expand(A, k)
    n = size(A, 2)
    K = k*n

    abar = k * sum(A, dims=2) / K
    C = k * (A .- abar) * (A .- abar)'

    t = tr(C)
    return (2π)^2/K * sqrt(2π/t)
end

begin
    ls = 0:0.1:5
    A = randn(3, 3)
    # A = A * sign(det(A))

    exacts = [Uintegral(l * A) for l in ls]
    expands = [Uintegral_expand(l * A) for l in ls]

    lines(ls, exacts ./ expands)
    ylims!(0, 2)
    current_figure()
end

function Uintegral_expand(A::AbstractMatrix)
    x, y, z = svdvals(A)
    z *= sign(det(A))

    # a = (σs[1] + σs[2]) / 2
    # b = (σs[1] - σs[2]) / 2
    # c = σs[3] * sdet

    return ϕd_approx([x, y, z])
end
function ϕd_approx(σs::AbstractVector)
    x, y, z = σs
    return (2π)^(3/2) * exp((x + y + z)) / sqrt((x+y)*(y+z)*(x+z))
end

function Uintegral(A::AbstractMatrix; kwargs...)
    σs = svdvals(A)
    sdet = sign(det(A))

    a = (σs[1] + σs[2]) / 2
    b = (σs[1] - σs[2]) / 2
    c = σs[3] * sdet
    f(x, p) = exp(c * x) * besseli(0, a*(x+1)) * besseli(0, b*(x-1))
    prob = IntegralProblem(f, (-1, 1))
    res = solve(prob, QuadGKJL(); kwargs...)
    return 4π^2 * res.u
end
function Uintegral_test(A; kwargs...)
    f(x, p) = exp(tr(A * R(x))) * sin(x[2])
    prob = IntegralProblem(f, zeros(3), [2π, π, 2π])
    res = solve(prob, HCubatureJL(); kwargs...)
    return res.u
end
function Rmean_test(A; kwargs...)
    f(x, p) = R(x) * exp(tr(A * R(x))) * sin(x[2])
    prob = IntegralProblem(f, zeros(3), [2π, π, 2π])
    res = solve(prob, HCubatureJL(); kwargs...)
    return res.u / Uintegral(A; kwargs...)
end

function outer(R)
    return [R[i,j] * R[k,l] for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
end
function Rvar_test(A; kwargs...)
    f(x, p) = outer(R(x)) * exp(tr(A * R(x))) * sin(x[2])
    prob = IntegralProblem(f, zeros(3), [2π, π, 2π])
    res = solve(prob, HCubatureJL(); kwargs...)
    return res.u / Uintegral(A; kwargs...)
end

function Rintegral_test(A; kwargs...)
    f(x, p) = R(x) * exp(tr(A * R(x))) * sin(x[2])
    prob = IntegralProblem(f, zeros(3), [2π, π, 2π])
    res = solve(prob, HCubatureJL(); kwargs...)
    return res.u
end
function Uintegral_exact_wrong(σs; kwargs...)
    α, β, γ = σs
    F32 = sinh(α) * sinh(β) * sinh(γ) / (α * β * γ)
    # F52 = 3 / (α * β * γ) * sum(e1*e2*e3 * sinh(e1*α + e2*β + e3*γ) / (e1*α + e2*β + e3*γ)^2 for e1 in [-1, 1], e2 in [-1, 1], e3 in [-1, 1])
    return 8π^2 * F32
end

function entropy3d(A, k)
    n = size(A, 2)
    K = k*n

    abar = k * sum(A, dims=2) / K
    C = k * (A .- abar) * (A .- abar)'

    σs = svdvals(C)
    t = tr(C)
    return 8π^2 * (2π/K)^(3/2) * exp(-t) * Uintegral(σs)
end

function entropy_fixed(energy_fn, r; tether=1e-12)
    D = length(r)
    x0 = D == 2 ? Float64[0, 0, 0, r..., 0] : Float64[0, 0, 0, 0, -π/2, 0, r..., 0, -π/2, 0]
    H = ForwardDiff.hessian(energy_fn, x0 + ones(length(x0)) * tether)

    start = D == 2 ? 4 : 7
    H = H[start:end, start:end]
    λs = eigvals(H)
    S_vib = -0.5 * sum(log, λs / (2π); init=0)

    χ = D == 2 ? 2π : 8π^2
    return χ * exp(S_vib)
end

function entropy_direct(A, k; V)
    D = size(A, 1)
    L = V^inv(D)
    energy, reshape_fn = make_dimerenergy(A; k)
    # f(x, p) = exp(-energy_fn(reshape([0, 0, 0, x...], 3, 2))) * 2π * π
    # prob = IntegralProblem(f, ([-L/2, -L/2, 0], [L/2, L/2, 2]))

    # f(x, p) = exp(-energy_fn(reshape(x, 3, 2))) / V * π^2
    # prob = IntegralProblem(f, ([-L/2, -L/2, 0, -L/2, -L/2, 0], [L/2, L/2, 2, L/2, L/2, 2]))

    # just separate off center of mass
    f(x, p) = exp(-energy(reshape_fn(x; dropfirst=true)...)) * sin(x[2]) * sin(x[8])

    bounds2d = ([0, -L/2, -L/2, 0], [2π, L/2, L/2, 2π])
    bounds3d = ([0, 0, 0, -L/2, -L/2, -L/2, 0, 0, 0], [2π, π, 2π, L/2, L/2, L/2, 2π, π, 2π])

    prob = IntegralProblem(f, D == 2 ? bounds2d : bounds3d)

    # prob = IntegralProblem(f, ([-L/2, -L/2, 1, -L/2, 0, 1], [L/2, L/2, 2, L/2, L/2, 2]))
    res = solve(prob, VEGASMC(niter=30, neval=2e6, print=1), reltol=1e-5, abstol=1e-5)
    return res.u
end


begin
    d = 0.5
    r = 1
    A = [d d;
        -r/2 r/2]
    k = 1
    r0 = vec(sum(A, dims=2)/size(A, 2))
end
begin
    energy_fn, reshape_fn = make_dimerenergy(A, A[:, [2, 1]]; k=5, offset=false)
    E(x) = energy_fn(reshape_fn(x)...)
    E([zeros(2); 0; 2r0; π])
    h = ForwardDiff.gradient(x->E(x), [zeros(2); 0; 2r0; π+0.1])
end

begin
    energy_fn, resh = make_dimerenergy(A, A[:, [2, 1]]; k)
    eflat = x->energy_fn(resh(x)...)
    Ω_truth = entropy_direct(A, k; V=10)

    Ω_approx_fix = entropy_fixed(eflat, [2d, 0])

    Ω_exact = entropy2d(A, k)
    Ω_expand = entropy2d_expand(A, k)

    Ω_approx_fix, Ω_expand, Ω_exact, Ω_truth
end

begin
    A3d = [d d d d;
           -r/2 r/2 0 0;
           0 0 -r/2 r/2]

    Ω_truth = entropy_direct(A3d, k; V=20)
end
begin
    energy_fn, resh = make_dimerenergy(A3d; k)
    eflat = x->energy_fn(resh(x)...)

    Ω_approx_fix = entropy_fixed(eflat, [d, 0, 0])
    Ω_exact = entropy3d(A3d, k)

    Ω_approx_fix, Ω_exact, Ω_truth
end




begin
    Jx = [0 0 0; 0 0 -1; 0 1 0]
    Jy = -[0 0 -1; 0 0 0; 1 0 0]
    Jz = [0 -1 0; 1 0 0; 0 0 0]

    Rexp(ψ::Euler) = exp(ψ[1] * Jz) * exp(ψ[2] * Jy) * exp(ψ[3] * Jz)
end







# Compare with ROly energy function
begin
    rules = [1 4 2 2]
    sys = AssemblySystem(rules, UnitSquareGeometry)

    strs = polygen(sys)
    s = strs[end]
end
begin
    e_old = Adiac.map_potentials(Adiac.twospring_bond, s, sys; ω=sqrt(k), r=r, ε=0)
    e_new, resh2d = make_dimerenergy([0.5 0.5; -0.5 0.5], [-0.5 -0.5; -0.5 0.5]; k=k)
end
begin
    ξ = Adiac.combinecoords(s.xs, s.ψs)

    e_old(ξ), e_new(resh2d(vec(ξ))...)
end

begin
    xs = -1:0.01:1.5
    ys = -1:0.01:1.5

    ϕ = 0
    ϕ2 = 0.5
    es = [exp(-e_new([x, y], Angle(ϕ), [d, 0], Angle(ϕ2))) for x in xs, y in ys]

    es_old = [exp(-e_old([x d; y 0; ϕ/π ϕ2/π])) for x in xs, y in ys]
    heatmap(xs, ys, es_old)

    f = Figure(size=(700, 300))
    ax = Axis(f[1, 1])
    ax2 = Axis(f[1, 2])
    h = heatmap!(ax, xs, ys, es)
    h2 = heatmap!(ax2, xs, ys, es_old)

    Colorbar(f[1, 1][1, 2], h)
    Colorbar(f[1, 2][1, 2], h2)

    ax.xticks = -1:0.5:2
    ax2.xticks = -1:0.5:2
    ax.yticks = -1:0.5:2
    ax2.yticks = -1:0.5:2

    current_figure()
end
begin
    svib, srot = entropy(s, sys; ε=10, ω=sqrt(k), r=r, atol=1e-6)
    svib_fix, srot_fix = entropy_fixed(s, sys; ε=10, ω=sqrt(k), r=r)

    Ω_approx = exp(svib+srot)
    Ω_approx_fix = exp(svib_fix+srot_fix)

    Ω_exact = entropy2d(A, k)
    Ω_expand = entropy2d_expand(A, k)

    Ω_approx, Ω_approx_fix, Ω_expand, Ω_exact
end


amean(A) = sum(A, dims=2) / size(A, 2)

function acov(A)
    abar = amean(A)
    C = (A .- abar) * (A .- abar)'
    return C
end

begin
    A = stack([x, 1] for x in -1:0.001:1)
end


function map_potential(bond_potential::Function, p::Polyform, sys::AssemblySystem)
    n = size(p)
    es = Adiac.exterior_edges(p.anatomy)
    bonds = ((Roly.vertex2particle(p, sys, e.src), Roly.vertex2particle(p, sys, e.dst)) for e in es)
    geoms = sys.geometries
    spcs = Roly.species(p)

    d = Roly.dimension(p)
    d == 2 || throw(ArgumentError("Can only handle 2d polyforms"))

    function energy_fn(ξs::AbstractMatrix{<:Real})
        size(ξs) == (3, n) || throw(ArgumentError("Invalid coordinates"))
        E = 0
        for ((i, si), (j, sj)) in bonds
            xi0, ψi0 = p.xs[i], p.ψs[i].θ
            xj0, ψj0 = p.xs[j], p.ψs[j].θ

            Δϕ = atan((xj0 - xi0)[2], (xj0 - xi0)[1])

            Ri = R(ψi0)
            Rj = R(ψj0)

            xi, ψi = @views ξs[1:d, i], ξs[d+1:end, i]
            xj, ψj = @views ξs[1:d, j], ξs[d+1:end, j]

            E += bond_potential(xi, inv(Angle(ψi0)) * Angle(ψi), xi + R(Δϕ)' * (xj - xi), inv(Angle(ψj0)) * Angle(ψj))
        end
        return E
    end

    return energy_fn
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

function entropy_laplace(energy_fn, ξ0; tether=1e-12)
    d, n = size(ξ0)
    d == 3 || throw(ArgumentError("Only works in 2d"))
    n > 1 || return 2π

    H = ForwardDiff.hessian(energy_fn, ξ0 .+ tether)[4:end, 4:end]
    λs = eigvals(H)
    S_vib = -0.5 * sum(log, λs / (2π); init=0)
    return 2π * exp(S_vib)
end

function entropy_dimer(A, k)
    n = size(A, 2)
    K = k*n

    abar = k * sum(A, dims=2) / K
    C = k * (A .- abar) * (A .- abar)'

    t = tr(C)
    return (2π)^3/K * exp(-t) * besseli(0, t)
end

function entropy_dimer_taylor(A, k)
    n = size(A, 2)
    K = k*n

    abar = k * sum(A, dims=2) / K
    C = k * (A .- abar) * (A .- abar)'

    t = tr(C)
    return (2π)^2/K * sqrt(2π/t)
end

function entropy_meanfield_meanz(np, nb; K, σ)
    d = nb / np
    return 2π * (320 * π^5 / (K^3 * σ^2 * (8d - 3)))^((np - 1) / 2) * exp(-2nb*(1-1/np))
end

function entropy_meanfield(np, nb; K, σ)
    d = nb / np
    return 2π * (320 * π^5 / (K^3 * σ^2 * (8d - 3)))^((np - 1) / 2) * exp(-2nb*(1-1/np))
end

begin
    rules = [1 1 1 3; 1 2 1 4]
    sys = AssemblySystem(rules, UnitSquareGeometry)

    strs = polygen(sys; maxsize=10)
    s = strs[findmax(s->length(Adiac.exterior_edges(s.anatomy)), strs)[2]]

    np = size(s)
    nb = length(Adiac.exterior_edges(s.anatomy))
end

begin
    n = 100
    σ = 1
    A = stack([σ/2, y] for y in range(-σ/2, σ/2; length=n))
    B = A .- [1, 0]
    k = 40 / n
    K = k * n
    energy_fn, reshape_fn = make_dimerenergy(A, B; k)

    en_s = map_potential(energy_fn, s, sys)
    ξ = combinecoords(s.xs, s.ψs)

    2π * (entropy_dimer(A, k)/(2π))^(np-1), 2π * (entropy_dimer_taylor(A, k)/(2π))^(np-1)
    entropy_laplace(en_s, ξ), entropy_meanfield(np, np; K, σ)
end
######

begin
    # rules = [1 3 2 1; 2 4 1 1] #; 3 2 1 4; 3 1 1 4; 3 3 1 4; 3 4 1 4]
    rules = [1 1 1 2]
    sys = AssemblySystem(rules, UnitSquareGeometry)

    strs = polygen(sys; maxsize=10)
    s = strs[end]
end

begin
    n = 100
    σ = 1.
    A = stack([σ/2, y] for y in range(-σ/2, σ/2; length=n))
    B = A .- [σ, 0]
    k = 10 / n
    K = k * n
    energy_fn, reshape_fn = make_dimerenergy(A, B; k)

    en_s = map_potential(energy_fn, s, sys)
    ξ = combinecoords(s.xs, s.ψs)

   (2π * (entropy_dimer(A, k) / (2π))^3 / 20) / entropy_laplace(en_s, ξ)
end

function entropy_MC(s, sys, A, B, k; V)
    n = size(s)
    D = size(A, 1)
    L = V^inv(D)
    L = 4
    Φ = 1

    _energy_fn, _ = make_dimerenergy(A, B; k)
    energy_fn = map_potential(_energy_fn, s, sys)
    reshape_coords(x) = reshape(x, 3, n-1)

    f(x, p) = let x=reshape_coords(x)
        exp(-energy_fn(hcat(zeros(eltype(x), 3), x)))
    end

    ξ0 = combinecoords(s.xs, s.ψs)
    ξ0 = ξ0[:, 2:end] .- ξ0[:, 1]

    bounds = (vec(ξ0) + repeat([-L/2, -L/2, -Φ], n-1), vec(ξ0) + repeat([L/2, L/2, Φ], n-1))
    prob = IntegralProblem(f, bounds)

    res = solve(prob, VEGASMC(niter=50, neval=3e6, print=1), reltol=1e-3, abstol=1e-3)
    return res.u
end