Cylinder 
===

N=2, tE=1, Re=50
---

 * too little time error
 * no tolcor -- ind1 for tol 2.-10 way too bad

N=2, tE=1, Re=80
---
 * no convergence

N=3, tE=2, Re=80
---
 * not converging for `Nts=512`
 
N=3, tE=2, Re=75
---
 * looks good -- let alone the rapid convergence rate - which gets better for better reference solutions. Its OK for checking `[512, 1024, 2048]` against `ref=8192`
 * no useful results for low `tol`
 * long simulation times

N=3, tE=.2, Re=60
---
 * good conv behavior for `[32, 64, 128, 256, 512]` vs  `4096`

    * Nts = 32, v_error = 0.000463863930878, p_error = 0.000526335262441
    * Nts = 64, v_error = 0.000208665203525, p_error = 0.000216946729061
    * Nts = 128, v_error = 9.88858138638e-05, p_error = 9.59969369052e-05
    * Nts = 256, v_error = 5.00126001464e-05, p_error = 4.53070079658e-05

 * but not so nice pics -- no vertices yet


Schedule for Experiments as of Jan 15th
===

Index 2 
---
 * Get it run on *Heinrich*
 * N=3, tE=.2, Re=60
 * One folder per test:

 1. `tol=2**(-14)`, no scaling of the momentum eq., `Ntsl=[32, 64, 128, 256, 512]`, `maxiter=tba`

    * too much an error in `p`

 1. `tol=2**(-18)`, no scaling of the momentum eq., `Ntsl=[32, 64, 128, 256, 512]`, `maxiter=800`
