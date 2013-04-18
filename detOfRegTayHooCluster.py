# use this script as 
#
#                       $python detOfRegTayHooCluster.py N
#
# where N > 2 is an integer defining the number
# of triangles in the patch to be considered

# This module checks invertibility of the local finite element matrix
# on a uniform cluster for a selection of velocity and
# pressure basis functions for the Taylor-Hood finite Elements

# N = number of triangles of the patch (N > 2)
# n = number of pressure nodes, that haven't been assigned 
#       to a different cluster before
# alp = parameter in [0,2*pi]

# assume n =< N, as and a patch has 
# either a neighbor considered before, or gets
# the single indeterminant pressure DOF substracted

# it has been shown, that for a uniform cluster invertibility
# depends only on one parameter alp, describing e.g. what angle the 
# edge associated to velBas_1 encloses with the x-axis.

# this script plots the determinant of the local finite element matrix
# versus the parameter alp for the choice of the velocity basis function
# according to the rule:

# Rule for choosing the velocity components
#
#                       let a_i be the angle edge i encloses with the x-axis

#                       if -pi/6  < a_i < pi/3   , take the y-component 
#                       if 5*pi/6 < a_i < 4*pi/3 , take the y-component 
#
#           else , take the x-component

# as the determinants are never zero, for any value of alp
# the choice of the basis velocity basis according to this algorithm
# renders the cluster matrix invertible

import numpy as np
from numpy import cos, sin, pi
import pylab as pl

def main(NU):

        ## Vectors of derivative directions
        # i.e. the choice of the velocity component
        # 0 <- x, 1 <- y 
        # to be corrected according to the rule 
        difDirVec = np.zeros((NU,1))

        # corrects the choice of the component, if by changing alpha
        # the edge enters the sector specified in the RULE
        corX = 1#None
        corY = 1#None

        #set this to None if there is no gap between the p-nodes on the periphery 
        gapinthenodes = None #1

        # Plots 
        myaxestyle = {
                'fontsize'            : 20 ,
                'weight'              : 'bold' ,
                'verticalalignment'   : 'top',
                'horizontalalignment' : 'center'
                }

        # number of discretization points 
        numPoints = 233

        #start with -pi/3, so that at alp=0 the choice is correct
        alphP = np.linspace(-pi/3,2*pi,num=numPoints)

        pl.close()
        pl.figure()
        pl.clf()

        for n in range(1,NU+1):
                detTH = np.zeros(numPoints)
                for j,alph0 in enumerate(alphP):
                        detTH[j] = comp_detslikerob(NU,n,alph0,thresh=0.5,corrX=corX,corrY=corY,difVec=difDirVec[:n,0],gitn=gapinthenodes)
                        # note that python calls by reference,
                        # thus the changes in difDirVec are kept
                        # till the next switch

                leg = '$n$ = $%d$' % n
                pl.plot(alphP,detTH,marker='o',label=leg)
                pl.xlabel('$a_1$',fontdict=myaxestyle)
                pl.ylabel('det',fontdict=myaxestyle)
                pl.xticks([0,pi,2*pi],['0','$\pi$','$2\pi$'])
                pl.xlim( 0, 2*pi+pi/4)
                pl.legend()
                if gapinthenodes is None:
                        pl.title('Cluster of $%d$ tets with $n$ considered pressure nodes' % NU)
                else:
                        pl.title('Det contribution of $n-2$ separated pressure nodes on a Cluster of $%d$ tets' % NU)
                pl.plot(alphP,0*alphP)

                #pl.show(block=False)   #use in [i]python shell, where the plot window 
                                                                #is not closed with function exit
        pl.show(block=True)

def comp_ni(N,n,i,alp):
        """computes the normal vector of the edge of triangle i
        associated with velocity bubble i 
        !ni points outside!  """
        gam = 2*pi/N
        return np.array([[-sin(alp + (i-1)*gam)], \
                        [cos(alp + (i-1)*gam)]])

def comp_mi(N,n,i,alp):
        """computes the normal vector of the outer edge of triangle i
        !mi points outside!  """
        gam = 2*pi/N
        return np.array([[-sin(alp + (i-1)*gam - 0.5*(pi-gam))], \
                        [cos(alp + (i-1)*gam - 0.5*(pi-gam))]])

def xyvec_int_pixdivvj(N,n,i,j,alp):
        """computes the integral of the pressure basis fctn pi

        times the div of vel basis vj
        i,j local enumeration on the cluster"""

        if i == 0:              #the center node
                return comp_ni(N,n,j-1,alp)-comp_ni(N,n,j+1,alp)
        else:
                if j == i:
                        return -comp_ni(N,n,i-1,alp)+comp_ni(N,n,i+1,alp)
                elif j == i-1:
                        return -comp_ni(N,n,i-1,alp)
                elif j == i+1:
                        return comp_ni(N,n,i+1,alp)
                else:           #no contribution outside the support
                        return np.zeros((2,1))

def comp_locTHmats(N,n,alp):
        """computes the FEM matrices of the local cluster

        for the x-component of the vel bubbles and
        the y-component of the vel bubbles
        """
        
        locTHmatX = np.zeros((n,n))
        locTHmatY = np.zeros((n,n))
        for i in range(n):
                for j in range(n):
                        contriVec = xyvec_int_pixdivvj(N,n,i,j,alp)
                        locTHmatX[i,j] = contriVec[0,0]
                        locTHmatY[i,j] = contriVec[1,0]
        if n == N:  #if all pressure nodes except from 1 are used
                                #additional entry in the left lower corner 
                                #since p_n and v_0 have a common support
                contriVec = xyvec_int_pixdivvj(N,n,n,n+1,alp)
                locTHmatX[n-1,0] = contriVec[0,0]
                locTHmatY[n-1,0] = contriVec[1,0]

        return locTHmatX, locTHmatY

def comp_detslikerob(N,n,alp,difVec=None,thresh=None,corrX=None,corrY=None,debu=None,gitn=None):

        #threshold for what edge is almost parallel to x-axis 
        if thresh is None:
                thresh = 0.5*np.sqrt(2)

        if difVec is None:
                difVec = np.ones(n)

        if corrX is not None:
                # the vector of angles the edges enclose with the x-axis
                gamVec = alp + 2*pi/N * np.arange(-1,n-1)
                # check distance to x-axis and grab the indices
                indVec = np.where( abs(sin(gamVec)) < thresh )
                # take y-comp there
                difVec[indVec] = 1

        if corrY is not None:
                # the vector of angles the edges enclose with the x-axis
                gamVec = alp + 2*pi/N * np.arange(-1,n-1)
                # to check distance to y-axis and grab the indices
                indVec = np.where( abs(cos(gamVec)) < thresh )
                # take x-comp there
                difVec[indVec] = 0

        decMatX = np.diag(difVec)
        decMatY = np.diag(1 - difVec)

        matX, matY = comp_locTHmats(N,n,alp)

        if debu is not None:
                return matX, matY, decMatX, np.linalg.det(np.dot(matX,decMatX) + np.dot(matY,decMatY))
        # to check the case when the 'active' pressure nodes have a gap at the border 
        elif gitn is not None:
                matX, matY, decMatX, decMatY = matX[1:,1:], matY[1:,1:], decMatX[1:,1:], decMatY[1:,1:]
                # we need at least 3 nodes to get a gap 
                if n < 2:
                        return 0
                else:
                        return np.linalg.det(np.dot(matX,decMatX) + np.dot(matY,decMatY))
        else: 
                return np.linalg.det(np.dot(matX,decMatX) + np.dot(matY,decMatY))

if __name__ == '__main__':
        from sys import argv, exit
        # input from command line
        if len(argv) == 2:
                script, NU = argv
                NU = int(NU)
        else:
                print "Specify the number of elements in the cluster!"
                exit(0)

        # exception handlings
        if NU < 3:
                print "Minimal number of patches is 3"
                exit(0)

        main(NU)
