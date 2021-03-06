import matplotlib.pyplot as plt
import numpy as np

import json


def load_json_dicts(StrToJs):

    fjs = open(StrToJs)
    JsDict = json.load(fjs)
    return JsDict


def merge_json_dicts(CurDi, DiToAppend):

    Jsc = load_json_dicts(CurDi)
    Jsa = load_json_dicts(DiToAppend)

    if Jsc['SpaceDiscParam'] != Jsa[
            'SpaceDiscParam'] or Jsc['Omega'] != Jsa['Omega']:
        raise Warning('Space discretization or omega do not match')

    Jsc['TimeDiscs'].extend(Jsa['TimeDiscs'])
    Jsc['ContiRes'].extend(Jsa['ContiRes'])
    Jsc['VelEr'].extend(Jsa['VelEr'])
    Jsc['PEr'].extend(Jsa['PEr'])
    # Jsc['TolCor'].extend(Jsa['TolCor'])

    JsFile = 'json/MrgdOmeg%dTol%0.2eNTs%dto%dMesh%d' \
        % (Jsc['Omega'],
           Jsc['LinaTol'],
           Jsc['TimeDiscs'][0],
           Jsc['TimeDiscs'][-1],
           Jsc['SpaceDiscParam']) + Jsc['TimeIntMeth'] + '.json'

    f = open(JsFile, 'w')
    f.write(json.dumps(Jsc))
    f.close()

    print 'Merged data stored in \n("' + JsFile + '")'

    return JsFile


def convpltjsd(Jsc):

    Jsc = load_json_dicts(Jsc)

    Mdict = {
        'HalfExpEulInd2': 'Ind2',
        'HalfExpEulSmaMin': 'Ind1',
        'Heei2Ra': 'Ind2ra'}
    JsFile = 'om%d' % Jsc['Omega'] + 'json/' + Mdict[Jsc['TimeIntMeth']] + \
        'Tol%1.1eN%d' % (Jsc['LinaTol'], Jsc['SpaceDiscParam']) + '.json'

    f = open(JsFile, 'w')
    f.write(json.dumps(Jsc))

    print 'Data stored in \n("' + JsFile + '")'

    return


def jsd_plot_errs(JsDict):

    JsDict = load_json_dicts(JsDict)

    plt.close('all')
    for i in range(len(JsDict['TimeDiscs'])):
        leg = 'NTs = $%d$' % JsDict['TimeDiscs'][i]
        plt.figure(1)
        plt.plot(JsDict['ContiRes'][i], label=leg)
        plt.title(JsDict['TimeIntMeth'] + ': continuity eqn residual')
        plt.legend()
        plt.figure(2)
        plt.plot(JsDict['VelEr'][i], label=leg)
        plt.title(JsDict['TimeIntMeth'] + ': Velocity error')
        plt.legend()
        plt.figure(3)
        plt.plot(JsDict['PEr'][i], label=leg)
        plt.title(JsDict['TimeIntMeth'] + ': Pressure error')
        plt.legend()

    plt.show()

    return


def jsd_calc_l2errs(JsDict, plot=False, ptikzfile=None):

    jsd = load_json_dicts(JsDict)
    timelength = jsd['TimeInterval'][1] - jsd['TimeInterval'][0]
    contresl, velerrl, perrl = [], [], []
    for i in range(len(jsd['TimeDiscs'])):
        dx = timelength / jsd['TimeDiscs'][i]
        contresl.append(np.sqrt(np.trapz(np.square(jsd['ContiRes'][i]),
                        dx=dx)))
        velerrl.append(np.sqrt(np.trapz(np.square(jsd['VelEr'][i]), dx=dx)))
        perrl.append(np.sqrt(np.trapz(np.square(jsd['PEr'][i]), dx=dx)))
    try:
        dconresl, momresl, tolcorl = [], [], []
        for i in range(len(jsd['TimeDiscs'])):
            dx = timelength / jsd['TimeDiscs'][i]
            dconresl.append(np.sqrt(np.trapz(np.square(jsd['DContiRes'][i]),
                            dx=dx)))
            momresl.append(np.sqrt(np.trapz(np.square(jsd['MomRes'][i]),
                           dx=dx)))
            tolcorl.append(np.sqrt(np.trapz(np.square(jsd['TolCor'][i]),
                           dx=dx)))

        allres = True
    except:
        allres = False
        print 'not all residuals were recorded'

    Ntsl = jsd['TimeDiscs']

    print 'L2 errors for method ' + jsd['TimeIntMeth']
    print 'N = ', jsd['SpaceDiscParam']
    print 'Nts = ', jsd['TimeDiscs']
    print 'Velocity Errors: ', velerrl
    print 'Pressure Errors: ', perrl
    print 'Conti Residuals: ', contresl
    if allres:
        print 'DConti Residuals: ', dconresl
        print 'Momenteq Residuals: ', momresl
        print 'TolCor: ', tolcorl
        if plot:
            plt.figure()
            plt.loglog(Ntsl, dconresl, 'o')
            plt.title('dconres')
            plt.figure()
            plt.loglog(Ntsl, momresl, 'v')
            plt.title('momres')
            plt.figure()
            plt.loglog(Ntsl, tolcorl, '^')
            plt.title('tolcors')

    if plot:
        plt.figure()
        plt.loglog(Ntsl, velerrl, 'o')
        plt.title('velerror')
        plt.figure()
        plt.loglog(Ntsl, perrl, 'v')
        plt.title('perror')
        plt.figure()
        plt.loglog(Ntsl, contresl, '^')
        plt.title('contres')
        if ptikzfile is None:
            plt.show(block=False)
    if ptikzfile is not None:
        import matlibplots.conv_plot_utils as cpu
        # see git@github.com:highlando/mat-lib-plots.git
        cpu.conv_plot(1./np.array(Ntsl[:-1]), [perrl[:-1]],
                      leglist=['perr'],
                      markerl=['*'], fit=[-1],
                      logscale=True, tikzfile=ptikzfile)
