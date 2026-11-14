# Cases of the throughput benchmarks: D + C + e on circular flux surfaces that differ in
# aspect ratio, safety factor, density, temperature and radial electric field, like the
# radii of a profile. Needs `nalpha`, `ntheta` and `nx` to be defined.

function circular_field(ntheta, eps; R0=3.0, q=2.0, B0=2.0, rho=0.5)
    a = eps * R0 / rho
    theta = range(0, 2π; length=ntheta + 1)[1:ntheta]
    Bmag = reshape(B0 ./ (1 .+ eps .* cos.(theta)), ntheta, 1)
    return field_from_boozer(rho, Bmag, 0.0, B0 * R0, 1 / q, π * a^2 * B0; R_major=R0, a_minor=a)
end

function case(k, n)
    u = (k - 0.5) / n                       # like a radius, from the core to the edge
    T, density = 4e3 * (1 - 0.9u), 8e19 * (1 - 0.8u)
    species = [from_scale_lengths(Deuterium, T, 0.8 * density, 1 + 4u, 0.5 + 2u),
        from_scale_lengths(Species(12, 6), T, density / 30, 1 + 4u, 0.5 + 2u),
        from_scale_lengths(Electron, T, density, 1 + 4u, 0.5 + 2u)]
    return DKEOperator(circular_field(ntheta, 0.05 + 0.3u; q=1 + 3u^2), UniformPitchAngleGrid(nalpha), MaxwellSpeedGrid(nx),
        species, 2e3 * (u - 0.3); p1="4z")
end

function build(n)
    ops = Vector{DKEOperator{Float64}}(undef, n)
    Threads.@threads for k in 1:n
        ops[k] = case(k, n)
    end
    return ops
end
