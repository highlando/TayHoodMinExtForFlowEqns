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

    * Nts = 32, verror = 0.000463863930878, perror = 0.000526335262441
    * Nts = 64, verror = 0.000208665203525, perror = 0.000216946729061
    * Nts = 128, verror = 9.88858138638e-05, perror = 9.59969369052e-05
    * Nts = 256, verror = 5.00126001464e-05, perror = 4.53070079658e-05

 * for index 1 

    * Nts = 32, vrrror = 0.000519916629373, perror = 0.053042803582
    * Nts = 64, vrrror = 0.000275647555997, perror = 0.0534501422269
    * Nts = 128, verror = 0.000167640303474, perror = 0.0536561766832
    * Nts = 256, verror = 0.000150202725962, perror = 0.0537398380109
    * Nts = 512, verror = 0.00370312691911, perror = 0.0531355398759

 * but not so nice pics -- no vertices yet


Schedule for Experiments as of Jan 15th
===

Index 1
---
 * `tol=2**(-14)` -- no convergence woe 
 * `tol=2**(-18)` -- looks cool - but goes nuts for `Nts=512`
 * `tol=2**(-22)` -- looks cool - but goes nuts for `Nts=512`

Index 2 
---
 * Get it run on *Heinrich*
 * N=3, tE=.2, Re=60
 * One folder per test:

 1. `tol=2**(-14)`, no scaling of the momentum eq., `Ntsl=[32, 64, 128, 256, 512]`, `maxiter=tba`

    * too much an error in `p`

 1. `tol=2**(-18)`, no scaling of the momentum eq., `Ntsl=[32, 64, 128, 256, 512]`, `maxiter=800`
