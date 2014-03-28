from comp_timeschemes_main import solve_euler_timedep
omega = 8
ntsl = [64, 128]

solve_euler_timedep(method=1, Omega=8, tE=None, Prec=None,
                    N=12, NtsList=ntsl, LinaTol=None, MaxIter=None,
                    UsePreTStps=None, SaveTStps=None, SaveIniVal=None)
