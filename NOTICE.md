# Yancc.jl Notice

Yancc.jl is a Julia port of [yancc](https://github.com/f0uriest/yancc) ("Yet Another
NeoClassical Code") by Rory Conlin, distributed under the MIT license. The physics
model, the numerical scheme and the verification cases are the work of Rory Conlin and
Matt Landreman (University of Maryland); the original copyright notice is retained in
`LICENSE`.

## Citation

If Yancc.jl contributes to an academic publication, please cite the paper that
describes the method:

    @article{Conlin2026yancc,
      author  = {Conlin, Rory and Landreman, Matt},
      title   = {yancc: a {GPU}-accelerated, differentiable solver for neoclassical
                 transport in tokamaks and stellarators},
      journal = {Plasma Physics and Controlled Fusion},
      volume  = {68},
      number  = {11},
      pages   = {115001},
      year    = {2026},
      doi     = {10.1088/1361-6587/aeb628}
    }

## Reference data

The files in `test/golden` are produced by `test/golden/generate.py` from yancc
v0.0.3. They include magnetic field arrays computed by yancc from the DSHAPE VMEC
equilibrium, and SFINCS results for that equilibrium, both from the `tests/data`
directory of the yancc repository.

## Development attribution

The Julia port is maintained by Project Torrey Pines
(https://github.com/ProjectTorreyPines).
