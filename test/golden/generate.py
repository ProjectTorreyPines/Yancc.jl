"""Generate reference data for Yancc.jl from the Python/JAX yancc.

Run with an environment where the original yancc (https://github.com/f0uriest/yancc)
is importable, on CPU:

    JAX_PLATFORMS=cpu python generate.py [YANCC_REPO] [NAME ...]

YANCC_REPO (default: the directory containing the imported yancc package) is only
needed for the DSHAPE VMEC and SFINCS files in its tests/data. NAME selects the files
to write (default: all). Writes .npz files next to this script. All state vectors use the Python layout: C order over
(species, speed, pitch, theta, zeta).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import yancc
from yancc import Field, LocalMaxwellian, MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc._collisions import (
    EnergyScattering,
    FieldParticleScattering,
    FokkerPlanckLandau,
    PitchAngleScattering,
    RosenbluthPotentials,
)
from yancc._finite_diff import fd2, fdbwd, fdfwd
from yancc._misc import DKEConstraint, DKESources, dke_rhs
from yancc._trajectories import DKE, DKEPitch, DKESpeed, DKETheta, DKEZeta
from yancc.solve import _make_dke_solution
from yancc.species import (
    Deuterium,
    Electron,
    GlobalMaxwellian,
    Hydrogen,
    Species,
    _coulomb_logarithm,
    _gamma_ab,
    _nuD_ab,
    _nupar_ab,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = (
    sys.argv[1]
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1])
    else os.path.dirname(os.path.dirname(os.path.abspath(yancc.__file__)))
)
ONLY = [a for a in sys.argv[1:] if not os.path.isdir(a)]
Carbon = Species(12, 6)
Helium = Species(4, 2)
A = lambda x: np.asarray(x, dtype=np.float64)


def save(name, data):
    path = os.path.join(HERE, name + ".npz")
    # zero-length arrays (e.g. no background species) are left out
    data = {k: np.asarray(v) for k, v in data.items() if np.asarray(v).size > 0}
    np.savez_compressed(path, **data)
    print(f"{name}.npz: {len(data)} arrays, {os.path.getsize(path) / 1e3:.0f} kB")


def test_vector(n):
    """Deterministic state vector, a closed form of the flat (Python) index."""
    i = np.arange(n, dtype=np.float64)
    return np.sin(0.37 * i + 0.2) * (1 + 0.3 * np.cos(0.0171 * i))


def species_data(species, prefix=""):
    return {
        prefix + "mass": A([s.species.mass for s in species]),
        prefix + "charge": A([s.species.charge for s in species]),
        prefix + "temperature": A([s.temperature for s in species]),
        prefix + "density": A([s.density for s in species]),
        prefix + "dTdrho": A([s.dTdrho for s in species]),
        prefix + "dndrho": A([s.dndrho for s in species]),
    }


def field_data(field):
    keys = [
        "rho", "B_sup_t", "B_sup_z", "B_sub_t", "B_sub_z", "Bmag", "sqrtg", "dBdt",
        "dBdz", "bdotgradB", "BxgradrhodotgradB", "Bmag_fsa", "B2mag_fsa", "Psi",
        "iota", "R_major", "a_minor", "B0", "I", "G", "psi_rho", "psi_r",
    ]
    out = {"field_" + k: A(getattr(field, k)) for k in keys}
    out["field_NFP"] = np.asarray(int(field.NFP))
    return out


# ---------------------------------------------------------------------------- grids
def grids():
    d = {}
    for nx in (3, 5, 6, 8):
        g = MaxwellSpeedGrid(nx)
        for k in ("x", "wx", "xvander", "Dx_pseudospectral", "D2x_pseudospectral"):
            d[f"speed{nx}_{k}"] = A(getattr(g, k))
        d[f"speed{nx}_gauge_idx"] = np.asarray(g.gauge_idx)
    for na in (9, 15, 16):
        g = UniformPitchAngleGrid(na)
        for k in ("alpha", "xi", "wxi"):
            d[f"pitch{na}_{k}"] = A(getattr(g, k))
    h = 0.3
    for n in (5, 7, 9, 16):
        f1 = jnp.ones(n)
        for bc in ("periodic", "symmetric"):
            for p in ("4d", "2d", "4z", "2z"):
                d[f"fdfwd_{p}_{bc}_{n}"] = A(jax.jacfwd(fdfwd)(f1, p, h=h, bc=bc))
                d[f"fdbwd_{p}_{bc}_{n}"] = A(jax.jacfwd(fdbwd)(f1, p, h=h, bc=bc))
            for p in (2, 4):
                d[f"fd2_{p}_{bc}_{n}"] = A(jax.jacfwd(fd2)(f1, p, h=h, bc=bc))
    d["fd_h"] = np.asarray(h)
    save("grids", d)


# -------------------------------------------------------------------------- species
def species_golden():
    sp = [
        LocalMaxwellian(Electron, 2.0e3, 5e19, -3.1e3, -4e19),
        LocalMaxwellian(Deuterium, 1.5e3, 4.4e19, -2.2e3, -3.5e19),
        LocalMaxwellian(Carbon, 1.4e3, 1e18, -2.0e3, -0.8e18),
    ]
    x = MaxwellSpeedGrid(5).x
    n = len(sp)
    d = species_data(sp)
    d["x"] = A(x)
    d["v_thermal"] = A([s.v_thermal for s in sp])
    d["F"] = A([s(x * s.v_thermal) for s in sp])
    d["lnlambda"] = A([[_coulomb_logarithm(a, b) for b in sp] for a in sp])
    for tag, ll in (("", None), ("_ll17", 17.0)):
        d["gamma" + tag] = A([[_gamma_ab(a, b, ll) for b in sp] for a in sp])
        d["nuD" + tag] = A(
            [[_nuD_ab(a, b, x * a.v_thermal, ll) for b in sp] for a in sp]
        )
        d["nupar" + tag] = A(
            [[_nupar_ab(a, b, x * a.v_thermal, ll) for b in sp] for a in sp]
        )
    assert d["nuD"].shape == (n, n, 5)
    save("species", d)


# ----------------------------------------------------------------------- collisions
def dummy_field():
    one = np.ones((1, 1))
    return Field(0.5, one, one, one, one, one, one, 1.0, 1.0, 10.0, 1.0)


def collisions():
    field = dummy_field()
    d = {}
    cases = {
        # tag: (species, background, nx, na, nL, coulomb_log)
        "a": (
            [
                LocalMaxwellian(Hydrogen, 4.95e4, 4.7e19, -6.6e4, -2.5e19),
                LocalMaxwellian(Electron, 1.5e4, 4.7e19, -2.0e4, -2.5e19),
            ],
            [LocalMaxwellian(Carbon, 3.0e4, 5e17, 0.0, 0.0)],
            3, 9, 8, None,
        ),
        "b": (
            [
                LocalMaxwellian(Deuterium, 2.0e3, 4.0e19, -1e3, -1e19),
                LocalMaxwellian(Carbon, 1.8e3, 1.0e18, -1e3, -1e18),
                LocalMaxwellian(Electron, 2.5e3, 4.6e19, -1e3, -1e19),
            ],
            [],
            6, 11, 8, 17.0,
        ),
        # more Legendre modes than pitch nodes: minimum norm pitch projection
        "c": (
            [LocalMaxwellian(Deuterium, 2.0e3, 4.0e19, -1e3, -1e19)],
            [],
            3, 7, 8, 17.0,
        ),
    }
    for tag, (sp, bg, nx, na, nL, ll) in cases.items():
        sg, pg = MaxwellSpeedGrid(nx), UniformPitchAngleGrid(na)
        pot = RosenbluthPotentials(sg, sp, nL=nL)
        CL = PitchAngleScattering(field, pg, sg, sp, bg, 4, False, coulomb_log=ll)
        CE = EnergyScattering(field, pg, sg, sp, bg, False, coulomb_log=ll)
        CF = FieldParticleScattering(field, pg, sg, sp, pot, False, coulomb_log=ll)
        C = FokkerPlanckLandau(field, pg, sg, sp, bg, pot, 4, False, coulomb_log=ll)
        p = f"{tag}_"
        d.update(species_data(sp, p))
        d.update(species_data(bg, p + "bg_"))
        d[p + "params"] = A([nx, na, nL, -1.0 if ll is None else ll])
        d[p + "ddGxlk"], d[p + "dHxlk"], d[p + "Hxlk"] = (
            A(pot.ddGxlk), A(pot.dHxlk), A(pot.Hxlk),
        )
        d[p + "Txi"], d[p + "Txi_inv"] = A(CF.Txi), A(CF.Txi_inv)
        d[p + "CL_D"], d[p + "CL_nus"] = A(CL._D), A(CL.nus)
        d[p + "CE_M"] = A(CE._M)
        d[p + "CF_C"], d[p + "CF_GHhat"] = A(CF.C), A(CF._GHhat)
        d[p + "exchange"] = A(C._exchange)
        if tag != "b":
            d[p + "block"] = A(C._velocity_block()[0])
    save("collisions", d)


# ------------------------------------------------------------------ full DKE cases
def dense_solve(Aop, B, C, rhs):
    n = rhs.size
    Amat = np.asarray(jax.jacfwd(Aop.mv)(jnp.zeros(n)))
    m = B.shape[1]
    K = np.block([[Amat, B], [C, np.zeros((m, m))]])
    y = np.linalg.solve(K, np.concatenate([rhs, np.zeros(m)]))
    res = np.linalg.norm(K @ y - np.concatenate([rhs, np.zeros(m)]))
    print(f"   dense solve N={n}: cond={np.linalg.cond(K):.2e} residual={res:.2e}")
    return y


OUTPUTS = [
    "<particle_flux>", "<heat_flux>", "<momentum_flux>", "<V||B>", "<J||B>", "J_rho",
    "particle_source", "heat_source",
]
KEY = {
    "<particle_flux>": "particle_flux", "<heat_flux>": "heat_flux",
    "<momentum_flux>": "momentum_flux", "<V||B>": "VparB", "<J||B>": "JparB",
    "J_rho": "J_rho", "particle_source": "particle_source",
    "heat_source": "heat_source",
}


def dke_case(field, sp, bg, nx, na, nL, ll, Erho, EparB, full=True, p1="4d"):
    sg, pg = MaxwellSpeedGrid(nx), UniformPitchAngleGrid(na)
    pot = RosenbluthPotentials(sg, sp, nL=nL)
    shape = (len(sp), nx, na, field.ntheta, field.nzeta)
    n = int(np.prod(shape))
    v = jnp.asarray(test_vector(n))
    args = (field, pg, sg, sp, Erho)
    ops = {
        "opx": DKESpeed(*args, False),
        "opa": DKEPitch(*args, p1, 4, False),
        "opt": DKETheta(*args, p1, 4, False),
        "opz": DKEZeta(*args, p1, 4, False),
        "dke": DKE(*args, bg, pot, p1, 4, False, None, ll),
    }
    d = {"params": A([nx, na, nL, -1.0 if ll is None else ll, Erho, EparB])}
    d.update(species_data(sp))
    d.update(species_data(bg, "bg_"))
    d.update(field_data(field))
    for k, op in ops.items():
        d["mv_" + k] = A(op.mv(v))
    Cop = ops["dke"]._C
    d["mv_CL"], d["mv_CE"], d["mv_CF"] = A(Cop.CL.mv(v)), A(Cop.CE.mv(v)), A(Cop.CF.mv(v))
    d["mv_C"] = A(Cop.mv(v))
    B = np.asarray(DKESources(field, pg, sg, sp).matrix)
    C = np.asarray(DKEConstraint(field, pg, sg, sp, True).matrix)
    rhs = np.asarray(dke_rhs(field, pg, sg, sp, Erho, EparB))
    if full:
        d["B"], d["C"] = B, C
    d["rhs"] = rhs
    y = dense_solve(ops["dke"], B, C, rhs)
    sol = _make_dke_solution(
        jnp.asarray(y), jnp.asarray(rhs), field, pg, sg, sp,
        jnp.asarray(Erho), jnp.asarray(EparB), bg,
    )
    if full:
        d["f1"] = y[:n]
    for q in OUTPUTS:
        d[KEY[q]] = A(sol.get(q))
    return d


def stellarator_field(nt=5, nz=5, NFP=5):
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)[:, None]
    zeta = np.linspace(0, 2 * np.pi / NFP, nz, endpoint=False)[None, :]
    Bmag = (
        2.50302 + 0.25660 * np.cos(NFP * zeta)
        - 0.11118 * np.cos(theta - NFP * zeta) - 0.05205 * np.cos(theta)
    )
    return Field.from_boozer(
        rho=0.5, Bmag=Bmag, I=0.0, G=14.4, iota=-0.88356, Psi=-2.0004,
        R_major=5.4832, a_minor=0.5211, NFP=NFP,
    )


def circular_tokamak_field(nt, eps, R0=1.7, q=2.0, B0=2.0, rho=0.5):
    a = eps * R0 / rho
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    Bmag = (B0 / (1 + eps * np.cos(theta)))[:, None]
    return Field.from_boozer(
        rho=rho, Bmag=Bmag, I=0.0, G=B0 * R0, iota=1 / q,
        Psi=np.pi * a**2 * B0, R_major=R0, a_minor=a, NFP=1,
    )


def dke3d():
    field = stellarator_field()
    mk = lambda s, T0: GlobalMaxwellian(
        s, lambda x: T0 * (1 - x**2), lambda x: 5e19 * (1 - x**4)
    ).localize(0.5)
    sp = [mk(Hydrogen, 6.60e4), mk(Electron, 2.00e4)]
    print("dke3d Erho=0")
    save("dke3d", dke_case(field, sp, [], 3, 9, 8, None, 0.0, 0.0))
    print("dke3d Erho=2000")
    save("dke3d_er", dke_case(field, sp, [], 3, 9, 8, None, 2.0e3, 0.0, full=False))


def tokamak():
    field = circular_tokamak_field(9, 0.3)
    mk = lambda s, T, n: LocalMaxwellian.from_scale_lengths(s, T, n, 3.0, 1.0)
    sp = [mk(Deuterium, 2.0e3, 3.6e19), mk(Carbon, 1.9e3, 1.5e18), mk(Electron, 2.2e3, 4.7e19)]
    bg = [mk(Helium, 2.0e3, 1.0e18)]
    print("tokamak")
    save("tokamak", dke_case(field, sp, bg, 4, 11, 8, 17.0, 0.0, 0.0))
    print("tokamak Erho, EparB")
    save("tokamak_er", dke_case(field, sp, bg, 4, 11, 8, 17.0, 300.0, 0.05, full=False))
    print("tokamak, centred stencil")
    save("tokamak_4z", dke_case(field, sp, bg, 4, 11, 8, 17.0, 300.0, 0.05, full=False, p1="4z"))


def dshape():
    wout = os.path.join(REPO, "tests", "data", "wout_DSHAPE.nc")
    d = {}
    for rho in (0.5, 0.8):
        f = Field.from_vmec(wout, rho, 15, 1)
        for k, v in field_data(f).items():
            d[f"rho{rho}_{k}"] = v
    # configuration of yancc's test_solve_dke_tokamak_axisymmetric, solved densely
    field = Field.from_vmec(wout, 0.5, 15, 1)
    sp = [
        LocalMaxwellian(
            Hydrogen, 0.8e3, 1.5e20, -2e3 * field.a_minor, -0.4e20 * field.a_minor
        )
    ]
    print("dshape")
    c = dke_case(field, sp, [], 5, 31, 8, 17.0, 0.0, 0.0, full=False)
    for k in ("params", "mass", "charge", "temperature", "density", "dTdrho", "dndrho"):
        d["axi_" + k] = c[k]
    for q in OUTPUTS:
        d["axi_" + KEY[q]] = c[KEY[q]]
    save("dshape", d)


# ------------------------------------------------- iterative solves at working size
def solve_case(field, sp, nx, na, Erho, ll, p1, rtol=1e-10, **kw):
    sol, info = yancc.solve_dke(
        field, UniformPitchAngleGrid(na), MaxwellSpeedGrid(nx), sp, Erho,
        coulomb_log=ll, rtol=rtol, p1=p1, **kw,
    )
    print(f"   N={len(sp) * nx * na * field.ntheta}: nmv={int(info['nmv'])} "
          f"res={float(info['res']):.1e} ok={bool(info['success'])}")
    assert bool(info["success"])
    d = {"params": A([nx, na, 8, ll, Erho, 0.0])}
    d.update(species_data(sp))
    d.update(field_data(field))
    for q in OUTPUTS:
        d[KEY[q]] = A(sol.get(q))
    return d


def production():
    """Circular tokamak, D + C + e, at a resolution used in practice."""
    eps, nustar, T, ll, zeff = 0.3, 1e-2, 2.0e3, 17.0, 2.0
    field = circular_tokamak_field(41, eps)
    # Sauter nu*_e = 6.921e-18 q R ne Zeff lnL / (Te^2 eps^1.5)
    ne = nustar * T**2 * eps**1.5 / (6.921e-18 * 2.0 * 1.7 * zeff * ll)
    nC = ne * (zeff - 1) / 30.0
    mk = lambda s, n: LocalMaxwellian.from_scale_lengths(s, T, n, 3.0, 1.0)
    sp = [mk(Deuterium, ne - 6 * nC), mk(Carbon, nC), mk(Electron, ne)]
    print("production")
    save("production", solve_case(field, sp, 6, 65, 0.0, ll, "4d"))


def sfincs_dshape():
    """DSHAPE tokamak radial scan of yancc's test_solve_dke_dshape_2species.

    SFINCS results (ntheta=71, nxi=160, nx=24) from the yancc repository, converted
    to the SI outputs of yancc; yancc solutions at two radii.
    """
    import interpax

    path = os.path.join(
        REPO, "tests", "data",
        "20260528-01-013_tokamak_radialScan_ntheta71_nxi160_nx24_tol1e-6_results.txt",
    )
    raw = np.loadtxt(path, delimiter="\t", skiprows=1)
    with open(path) as fh:
        head = [h.strip() for h in fh.readline().split("\t")]
    sf = {h: c for h, c in zip(head, raw.T)}
    rho, idx = np.unique(sf["rN"], return_index=True)
    spline = lambda k, scale: interpax.CubicSpline(sf["rN"][idx], scale * sf[k][idx])
    electrons = GlobalMaxwellian(Electron, spline("THats_electron", 1e3), spline("nHats_electron", 1e20))
    ions = GlobalMaxwellian(Hydrogen, spline("THats_ion", 1e3), spline("nHats_ion", 1e20))
    wout = os.path.join(REPO, "tests", "data", "wout_DSHAPE.nc")

    # SFINCS normalization: Rbar = 1 m, Bbar = 1 T, nbar = 1e20 m^-3, Tbar = 1 keV, mbar = m_p
    nbar, mbar = 1e20, 1.67262192595e-27
    vbar = np.sqrt(2 * 1e3 * 1.602176634e-19 / mbar)
    d = {"rho": A(rho)}
    print("sfincs_dshape")
    for ir, r in enumerate(rho):
        field = Field.from_vmec(wout, r, 71, 1)
        sp = [electrons.localize(r), ions.localize(r)]
        Erho = float(sf["Er"][ir] * 1000 * field.a_minor)
        c = {"Erho": A(Erho)}
        c.update(species_data(sp))
        c.update(field_data(field))
        # fluxes of rho = r/a from the fluxes of rHat [m]
        both = lambda k: np.array([sf[k + "_electron"][ir], sf[k + "_ion"][ir]])
        c["sfincs_particle_flux"] = both("particleFlux_vm_rHat") * nbar * vbar / field.a_minor
        c["sfincs_heat_flux"] = both("heatFlux_vm_rHat") * nbar * vbar**3 * mbar / field.a_minor
        c["sfincs_VparB"] = both("FSABFlow") * vbar * nbar / A([s.density for s in sp])
        c["sfincs_JparB"] = A(sf["FSABjHat"][ir] * 1.602176634e-19 * nbar * vbar)
        if ir in (15, 17):
            y = solve_case(field, sp, 12, 81, Erho, 17.0, "4d", rtol=1e-9)
            for q in OUTPUTS:
                c["yancc_" + KEY[q]] = y[KEY[q]]
        for k, v in c.items():
            d[f"ir{ir}_{k}"] = v
    save("sfincs_dshape", d)


if __name__ == "__main__":
    print("yancc", yancc.__version__, "jax", jax.__version__)
    for case in (grids, species_golden, collisions, dke3d, tokamak, dshape, production, sfincs_dshape):
        if not ONLY or case.__name__ in ONLY:
            case()
