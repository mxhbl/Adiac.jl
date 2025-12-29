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


function make_dimerenergy(As, Bs=As.-sum(As, dims=2)/size(As, 2); k)
    # the offset is needed for euler angles to avoid having equilibrium at gimbal lock
    # right now, this breaks 2d
    D = size(As, 1)

    R_offset = D == 2 ? I : R([0, π/2, 0])

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

function Fintegral(σs; kwargs...)
    a = (σs[1] + σs[2]) / 2
    b = (σs[1] - σs[2]) / 2
    c = σs[3]
    f(x, p) = exp(c * x) * besseli(0, a*x) * besseli(0, b*(1-x))
    prob = IntegralProblem(f, (0, 1))
    res = solve(prob, QuadGKJL(); kwargs...)
    return res.u
end

function Fintegral_exact(σs; kwargs...)
    a = (σs[1] + σs[2]) / 2
    b = (σs[1] - σs[2]) / 2
    c = σs[3]
    Δ = sqrt(complex((c^2 - (a^2 + b^2)) * (c^2 - (a^2 - b^2))))
    return exp(c/2) / Δ * sinh(Δ/2)
end



function Uintegral(σs; kwargs...)
    a = (σs[1] + σs[2]) / 2
    b = (σs[1] - σs[2]) / 2
    c = σs[3]
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

    energy_fn, resh = make_dimerenergy(A; k)
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