import plot_utils as plu

mrglist1 = [
    "json/Omeg8Tol2.44e-04NTs32to64Mesh40HalfExpEulInd2.json",
    "json/Omeg8Tol2.44e-04NTs128to128Mesh40HalfExpEulInd2.json",
    "json/Omeg8Tol2.44e-04NTs512to512Mesh40HalfExpEulInd2.json"
    ]

mrglist2 = [
    "json/Omeg8Tol2.44e-04NTs128to128Mesh40HalfExpEulSmaMin.json",
    "json/Omeg8Tol2.44e-04NTs32to64Mesh40HalfExpEulSmaMin.json",
    "json/Omeg8Tol2.44e-04NTs512to512Mesh40HalfExpEulSmaMin.json"
    ]

curdi = mrglist1[0]
for mrgdi in mrglist1[1:]:
    curdi = plu.merge_json_dicts(curdi, mrgdi)

plu.jsd_calc_l2errs(curdi, plot=True)

curdi = mrglist2[0]
for mrgdi in mrglist2[1:]:
    curdi = plu.merge_json_dicts(curdi, mrgdi)

plu.jsd_calc_l2errs(curdi, plot=True)
