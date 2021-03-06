Rothe
===
### smmx exps
 * with Nswitches --- smmx much worse than ind2 in veloctity
 * no jump in pressure
 * but jump in velocity -- however independent of `dt`


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

Index 1
---
 * `tolCor` - if set `False`, then the abs residual is to large:
 
L2 errors for method HalfExpEulSmaMin
N =  40
Nts =  [32, 64, 128], tE=.25
Velocity Errors:  [0.00010250621362684706, 8.6274789956598556e-05, 0.0001009061483171032]
Pressure Errors:  [3.2014034739937247e-05, 4.8826779718124515e-05, 8.4881943342043927e-05]
Conti Residuals:  [9.9115593408841389e-07, 1.257300751363873e-06, 1.8069006627940962e-06]
DConti Residuals:  [1.2473160904929237e-06, 2.256846312685112e-06, 4.1201417672178469e-06]
Momenteq Residuals:  [8.222777765777421e-09, 1.5481167346648894e-08, 3.1360399769489095e-08]

 * if `True` -- then the contiresidual goes down with `dt` -- try with unscaled `tq1` although this should not effect the residuals

Nts =  [32, 64, 128, 256, 512], tE=.25
Velocity Errors:  [9.6157254929077652e-05, 6.3502413359510428e-05, 5.2506982730858687e-05, 4.9147106591150811e-05, 4.8263184597584799e-05]
Pressure Errors:  [1.8051515735763941e-05, 2.0171541966270379e-05, 2.3874136499325368e-05, 2.4919083743043529e-05, 2.5704342982791795e-05]
Conti Residuals:  [2.4571883106152445e-07, 1.2839931918497056e-07, 8.794653014022853e-08, 3.4760014491852636e-08, 1.7408376211771105e-08]
DConti Residuals:  [1.4888559371880945e-07, 1.3841060113013898e-07, 1.3821053363067968e-07, 1.1114231363069306e-07, 9.4623780983134754e-08]
Momenteq Residuals:  [8.0992282878877124e-10, 8.2235423530079193e-10, 7.9837348801390103e-10, 7.6541966088369531e-10, 7.4770324021751187e-10]
TolCor:  [0.12181155388250441, 0.086022649096692008, 0.060713605684374748, 0.042872686939275061, 0.03027372143937886]

 * unscaled: Conti again goes down - because of TolCor?? -- gonna scale the dconti part in tolcor by `dt`

L2 errors for method HalfExpEulSmaMin
N =  40, tE=1.
Nts =  [64, 128, 256, 512, 1024], tE=1.
Velocity Errors:  [0.00056211041931751137, 0.00031149873971137105, 0.00020523500447589441, 0.00016777806358093516, 0.00015666206171642633]
Pressure Errors:  [3.6470596455454961e-05, 3.4943771823407833e-05, 4.073785264962144e-05, 4.7098427332802745e-05, 5.2338162665218873e-05]
Conti Residuals:  [1.4060191632436979e-05, 9.9894696675409381e-06, 5.5423152450290638e-06, 3.6575248529142277e-06, 4.1362468742419303e-06]
DConti Residuals:  [1.358216322468195e-06, 1.210851448519757e-06, 8.8765752698366314e-07, 8.0636696546996748e-07, 7.1565095157195322e-07]
Momenteq Residuals:  [1.3931811724143975e-09, 1.4017401906513117e-09, 1.4952918303716212e-09, 1.4528762491594132e-09, 1.4652338424853883e-09]
TolCor:  [0.41904121618788426, 0.29174646938217569, 0.20530498165534569, 0.13347472880881125, 0.088015710483781207]

 * looks a bit better - but still conti res gets worse and pressure similarly -- gonna try scaling by dt**0.5
 
L2 errors for method HalfExpEulSmaMin
N =  40
Nts =  [128, 256, 512]
Velocity Errors:  [0.00035449939234255534, 0.00029217308955105761, 0.00023526764928050688]
Pressure Errors:  [0.00011261681324546574, 0.00014225733897904047, 0.00020255717510153973]
Conti Residuals:  [2.7197164508351915e-05, 3.6684475701633219e-05, 6.5185057695600501e-05]
DConti Residuals:  [6.5654929039227664e-06, 6.8829033297068239e-06, 8.4297384422351603e-06]
Momenteq Residuals:  [1.1491529614563529e-08, 1.4936067017457968e-08, 1.805171397399415e-08]
TolCor:  [0.87443568883512912, 0.70272952927213161, 0.53914886161321574]

L2 errors for method HalfExpEulSmaMin, inner product for tolcor mit dt=sqrt(dt)
N =  40, tE=1.
Nts =  [128, 256, 512]
Velocity Errors:  [0.00032158538891832934, 0.00021721285067609526, 0.00018533548127640568]
Pressure Errors:  [5.471479570531747e-05, 7.3395216865098658e-05, 8.4168765453430971e-05]
Conti Residuals:  [2.1350357722715226e-05, 1.5769824470403889e-05, 3.3657068618072426e-05]
DConti Residuals:  [2.9161415245951878e-06, 2.8755298578401648e-06, 3.2992266314171492e-06]
Momenteq Residuals:  [4.6352763889911966e-09, 5.4296167877632185e-09, 6.294785666812842e-09]
TolCor:  [0.53988421809013476, 0.41468844549831885, 0.31550853313462585]
