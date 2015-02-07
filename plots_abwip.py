import plot_utils as plu
filelist = [
    "json/Omeg8Tol2.44e-04NTs16to256Mesh40HalfExpEulInd2." +
    "json_globalcount_kinivold",
    "json/Omeg8Tol2.44e-04NTs16to256Mesh40HalfExpEulInd2." +
    "json_globalcount_kinivupd",
    "json/Omeg8Tol2.44e-04NTs16to256Mesh40HalfExpEulInd2." +
    "json_globalcount_kinivzero",
    "json/Omeg8Tol2.44e-04NTs16to256Mesh40HalfExpEulSmaMin." +
    "json_globalcount_kinivold",
    "json/Omeg8Tol2.44e-04NTs16to256Mesh40HalfExpEulSmaMin." +
    "json_globalcount_kinivupd",
    "json/Omeg8Tol2.44e-04NTs512to512Mesh40HalfExpEulInd2." +
    "json_globalcount_kinivupd"
    ]
# "json/Omeg8Tol2.44e-04NTs16to256Mesh40HalfExpEulSmaMin." +
# "json_globalcount_kinivzero"]

for JsFile in filelist:
    # plu.jsd_plot_errs(JsFile)
    print '\n\n ###'
    print JsFile
    plu.jsd_calc_l2errs(JsFile)
    plu.jsd_count_timeiters(JsFile)
