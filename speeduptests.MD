Speedups
========

Sem Imp Eul (N=40)
------------------
 * ~10% per iteration by using factorized mass mats for the `M.-1` inner product (rather than CG)
 * `dt=2/16`: Iteration counts for the first 6 Time Steps
   * `x0=old`: 			(4, 732, 718, 694, 686, 726) = -152
   * `x0=old+dt*dx`: 	(4, 732, 690, 700, 710, 718) = -142
   * `x0=0`: 			(4, 732, 738, 740, 740, 742) = +0
 * `dt=2/128`: Iteration counts for the first 6 Time Steps
   * `x0=old+dt*dx\dt`:	(4, 686, 182, 160, 182, 272) no comment
   * `x0=old`: 			(4, 686, 676, 670, 662, 650)
   * `x0=0`: 			(4, 686, 692, 696, 696, 698)

Sma Min Sem Imp (N=40):
 * `dt=2/128`, Max It = 800: Iteration counts for the first 6 Time Steps
   * `x0=old+dt*dx`: 	(245, 249, 115, 104, 108, 115) 
   * `x0=old`: 			(245, 217, 233, 239, 240, 242) 

 * x0=xold; Restart=800 vs. Restart=200 vs. Restart = 100
   * see above 3. time step 233 of 800 iterations = 69 sec
   * see above 3. time step 318 of 200 iterations = 75 sec
   * see above 3. time step 315 of 150 iterations = 55 sec
   * see above 3. time step 533 of 100 iterations = 69 sec

 * x0=xold+dt*dx; Restart 800 vs. Restart = 100 vs. Restart 75
   * see above 3. time step 115 of 800 iterations = ?
   * see above 3. time step 167 of 100 iterations = 20 sec
   * see above 3. time step 191 of 75 iterations = 17 sec
   * see above 3. time step 149 of 85 iterations = 16 sec


