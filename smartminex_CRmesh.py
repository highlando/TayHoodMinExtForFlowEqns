import numpy as np
from dolfin import Mesh, cells

def get_smamin_rearrangement(N,PrP,Mc,Bc):
	from smamin_utils import col_columns_atend
	from scipy.io import loadmat, savemat

	# get the indices of the B2-part
	B2Inds = get_B2_bubbleinds(N, PrP.V, PrP.mesh)
	# the B2 inds wrt to inner nodes
	# this gives a masked array of boolean type
	B2BoolInv = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], B2Inds)
	# this as indices
	B2BI = np.arange(len(B2BoolInv), dtype=np.int32)[B2BoolInv]

	dname = '%sSmeMcBc' % N

	try: SmDic = loadmat(dname)

	except IOError:
		print 'Computing the B2 indices...'
		# get the indices of the B2-part
		B2Inds = get_B2_bubbleinds(N, PrP.V, PrP.mesh)
		# the B2 inds wrt to inner nodes
		# this gives a masked array of boolean type
		B2BoolInv = np.in1d(np.arange(PrP.V.dim())[PrP.invinds], B2Inds)
		# this as indices
		B2BI = np.arange(len(B2BoolInv), dtype=np.int32)[B2BoolInv]
		# Reorder the matrices for smart min ext...
		# ...the columns
		print 'Rearranging the matrices...'
		# Reorder the matrices for smart min ext...
		# ...the columns
		MSmeC = col_columns_atend(Mc, B2BI)
		BSme = col_columns_atend(Bc, B2BI)
		# ...and the lines
		MSmeCL = col_columns_atend(MSmeC.T, B2BI)
		print 'done'

		savemat(dname, { 'MSmeCL': MSmeCL, 'BSme':BSme, 
			'B2Inds':B2Inds, 'B2BoolInv':B2BoolInv, 'B2BI':B2BI} )
	
	SmDic = loadmat(dname)

	MSmeCL = SmDic['MSmeCL']
	BSme = SmDic['BSme']
	B2Inds = SmDic['B2Inds']
	B2BoolInv = SmDic['B2BoolInv']>0
	B2BoolInv = B2BoolInv.flatten()
	B2BI = SmDic['B2BI']

	return MSmeCL, BSme, B2Inds, B2BoolInv, B2BI 

