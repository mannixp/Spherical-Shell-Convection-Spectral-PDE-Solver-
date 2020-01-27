#! /usr/bin/env python

import numpy as np
import sys, time
from EigenVals import *

#from scipy.sparse import csr_matrix
#from Projection	import *

# Time-Stepping Routines
#@profile
def Matrices_Tstep(Pr,Ra,sigma,D,R,N_modes,dt,Re_1,Re_2): # Correct

	M = M_0_SP(D,R,N_modes);
	M_INV = np.linalg.inv(M);

	L0 = L_0_SP(D,R,sigma,Pr,Ra,N_modes);
	L_Omega = L_OM_SP(D,N_modes,R,sigma,Pr,Re_2,Re_1);
	L_Cor = L_COR_SP(D,N_modes,R,sigma,Pr,Re_2,Re_1);

	FF = force_SP(Pr,R,N_modes,sigma,Re_2,Re_1); 

	'''
	M = M_0(R,N_modes);
	M_INV = np.linalg.inv(M);

	LL = L_0(Pr,Ra,sigma,R,N_modes); # Thermal Part 
	L_Omega  = L_OM(N_modes,R,sigma,Pr,Re_2,Re_1); # Angualr Momentum Part
	L_Cor = L_COR(N_modes,R,sigma,Pr,Re_2,Re_1); # Coriolis term vort Part
	#LM = LL + L_Omega + L_Cor; # Their sum
	'''

	L = np.matmul(M_INV,L0 + L_Omega + L_Cor);
	F = np.matmul(M_INV,FF);

	return M_INV,L,F

#@profile
def IMPLICIT(RUNS,l,Pr,Ra_c,epsilon,sigma,Nr,N_modes,dt,Re_1,Re_2,XX): # Correct
	
	D,R = cheb_radial(Nr,sigma)

	Ra = Ra_c*(1.0 + epsilon);
	# Crank Nicholson Method, Correct
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~~~
	M_INV,L,FF = Matrices_Tstep(Pr,Ra,sigma,D,R,N_modes,dt,Re_1,Re_2); # Obtain the linear and forcing matrices

	L_left = np.eye(L.shape[0]) - 0.5*dt*L; L_right = np.eye(L.shape[0]) + 0.5*dt*L;
	L_l_INV = np.linalg.inv(L_left);
	
	L_CR = np.matmul(L_l_INV,L_right);
	N_CR = dt*np.matmul(L_l_INV,M_INV)
	F = dt*np.matmul(L_l_INV ,FF)
	N_Prev = np.zeros(F.shape)

	# Declare All matrices dependent on R
	nr = len(R)-2;
	MR2 = np.diag(1.0/(R[1:-1]**2)); #MR2 = np.matmul(MR2,D[1:-1,1:-1])
	MR3 = np.diag(1.0/(R[1:-1]**3)); 
	D3 = []; D2 = [];
	for ii in xrange(N_modes):
		#DD3 = np.matmul(MR2, D3_SP(D,R,ii));
		#D3.append(DD3);
		D3.append(D3_SP(D,R,ii));
		D2.append(Stokes_D2_SP(D,R,ii));	
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~~~~~~~~~~~~~~	

	# Declare Vector X	
	NN = 3*nr*N_modes;
	iters = RUNS;
	CAP_STEP = 5; #200;	
	tsteps = iters/CAP_STEP;
	X_VID = np.zeros((NN,tsteps));
	X = np.zeros(NN); Y = np.zeros(NN);
	X_new = np.zeros(X.shape);
	
	# Recast X if required
	if len(X) > len(XX):
		X[0:len(XX)] = XX[:]; #.reshape(len(XX));
	elif len(X) < len(XX):
		X[:] = XX[0:len(X)]; #.reshape(len(X));	
	else:
		X = XX;	
	
	err = 1.0; it = 0; ii = 0; X_Norm = [];
	start_time = time.time()
	while ii < iters: # err > 1e-08: #(1e-4):	

		#N_0 = np.matmul(M_INV,N_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B))
		#X_new = X + dt*np.matmul(L + N_0,X) + dt*FF; #  Euler explicit
		
		# Crank Nicholson, Correct Implemented
		N_Curr = np.matmul( N_Full(X,L.shape,N_modes,nr,MR2,MR3,D3,D2,D), X);		
		N_0 = 3.0*N_Curr - N_Prev;
		X_new = 0.5*np.matmul(N_CR,N_0) + np.matmul(L_CR,X) + F;
		N_Prev = N_Curr;

		err = np.linalg.norm((X_new - X),2)/np.linalg.norm(X_new,2)				
		print 'Error ',err
		if err > 100.0:
			print "Max error exceeded"
			break
		print 'Iteration: ', ii,"\n"

		EVE = 0.0; ODD = 0.0;
		for m in xrange(N_modes):
			if m == 0:
				EVE += np.linalg.norm(X[(len(R)-2):2*(len(R)-2),0],2);
			elif m == 1:
				ODD += np.linalg.norm(X[3*(len(R)-2):5*(len(R)-2),0],2);
			elif m%2 == 0:
				ind = 5*(len(R)-2) + (m-2)*3*(len(R)-2);
				EVE += np.linalg.norm(X[ind:ind+3*(len(R)-2),0],2);
			elif m%2 == 1:
				ind = 5*(len(R)-2) + (m-2)*3*(len(R)-2);
				ODD += np.linalg.norm(X[ind:ind+3*(len(R)-2),0],2);	
				
		print "EVEN ||X|| ",EVE,"\n"
		print "ODD  ||X|| ",ODD,"\n"

		X_Norm.append(np.linalg.norm(X));
		
		if (it<(tsteps-1)) and (ii%CAP_STEP == 0): 
			#------- ARRAY CAPTURE FOR VIDEO -------
			print "Captured frame n= ",it,"\n"
			X_VID[:,it] = np.squeeze(X_new, axis=1)[:]; 
			it+=1
		
		ii+=1;	
		X = X_new;	

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))		
	
	# --------------------------- SAVING AND COMPILING RESULTS -------------------
	if True:
		#1) Change directory into Results file
		#os.chdir('/home/mannixp/Dropbox/Imperial/MResProject/Python_Codes/Transmission/')	
		#os.chdir('/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/Pseudo_SPEC/')
		#os.chdir('~/')
		#Stringy = ['/home/ma/p/pm4615/Pseudo_SPEC/']
		#os.chdir("".join(Stringy))

		# 2) Make new results folder
		st_Pr = "%4.2f"%Pr; st_eps = "%2.2f"%epsilon;
		st_Re1 = "%4.1f"%Re_1; st_Re2 = "%4.1f"%Re_2;
		Stringy = ['Time_Dependent_1C_Sol_Pr',st_Pr,'_epsilon',st_eps,'_Re1',st_Re1,'_Re2',st_Re2];
		os.mkdir("".join(Stringy))

		# 3) Change into Results folder
		os.chdir("".join(Stringy))	

		#Create an associated Parameters file
		Parameters = np.zeros(18)
		file = open("Parameters.txt",'w')
		
		file.write(" #~~~~~~~~~~~~~~~~ Control Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

		Parameters[0] = Ra_c
		file.write('%5.4f'% Ra_c); file.write(" # Ra_c Critical Rayleigh #0 \n");
		
		Parameters[1] = sigma
		file.write('%5.3f'% sigma); file.write(" # d Sigma Gap Width #1 \n")

		Parameters[2] = Re_1
		file.write('%5.4f'% Re_1); file.write(" # Re_1 Inner Reynolds Number #2 \n" )
		Parameters[3] = Re_2
		file.write('%5.4f'% Re_2); file.write(" # Re_2 Outer Reynolds Number #3 \n" )

		Parameters[4] = epsilon
		file.write('%5.4f'% epsilon); file.write(" # epsilon #4 \n")
		
		file.write(" #~~~~~~~~~~~~~~~~ Numerical Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

		Parameters[5] = dt
		file.write('%1.6f' % dt); file.write(" # Time step #5 \n")

		Parameters[6] = len(R)
		file.write('%d' % len(R)); file.write(" # Nr Radial steps #6 \n")
		
		Parameters[7] = N_modes
		file.write('%d' % N_modes);	file.write(" # N_Modes Polar Mode #7 \n")

		Parameters[8] = Pr
		file.write('%5.4f'% Pr); file.write(" # Pr Prandtl nunber #8\n");	

		STR = "".join(['XD_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy'])
		STRVID = "".join(['XD_VID_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy'])

		np.save(STRVID,X_VID)
		np.save(STR,X)
		np.save("PSI_L2.npy",X_Norm)
		np.save("Parameters.npy",Parameters)

		#os.chdir('/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/Pseudo_SPEC/')
	
	return X,X_VID;	

# Steady-State Solving Routines	
def Matrices(D,Pr,Ra,sigma,R,N_modes,Re_1,Re_2):

	L0 = L_0_SP(D,R,sigma,Pr,Ra,N_modes);
	L_Omega = L_OM_SP(D,N_modes,R,sigma,Pr,Re_2,Re_1);
	L_Cor = L_COR_SP(D,N_modes,R,sigma,Pr,Re_2,Re_1);
	L = L0 + L_Omega + L_Cor;

	FF = force_SP(Pr,R,N_modes,sigma,Re_2,Re_1);	

	return L,FF

def Netwon_It(Pr,Ra,sigma,Nr,N_modes,Re_1,Re_2,XX): #'GF = 0.0): # Correct

	D,R = cheb_radial(Nr,sigma)
	L,F = Matrices(D,Pr,Ra,sigma,R,N_modes,Re_1,Re_2); # Obtain the linear and forcing matrices
	
	# Declare All matrices dependent on R
	nr = len(R)-2;
	MR2 = np.diag(1.0/(R[1:-1]**2)); #MR2 = np.matmul(MR2,D[1:-1,1:-1])
	MR3 = np.diag(1.0/(R[1:-1]**3)); 
	D3 = []; D2 = [];
	for ii in xrange(N_modes):
		#DD3 = np.matmul(MR2, D3_SP(D,R,ii));
		#D3.append(DD3);
		D3.append(D3_SP(D,R,ii));
		D2.append(Stokes_D2_SP(D,R,ii));

	# Declare Vector X	
	NN = 3*nr*N_modes;
	X = np.zeros((NN,1)); 
	X_new = np.zeros(X.shape);
	
	# Recast X if required
	if len(X) > len(XX):
		X[0:len(XX),0] = XX[:,0].reshape(len(XX));
	elif len(X) < len(XX):
		X[:,0] = XX[0:len(X),0].reshape(len(X));	
	else:
		X = XX;	
	
	err = 1.0; it = 0;
	
	start_time = time.time()
	while err > (1e-9):	
					
		N1 = N_Full(X,L.shape,N_modes,nr,MR2,MR3,D3,D2,D);
		N2 = N2_Full(X,L.shape,N_modes,nr,MR2,MR3,D3,D2,D);
		
		F_X = np.matmul(L+N1,X) + F; # L.multiply(X) + F + np.matmul(N1,X); Possible Sparse Speedup

		X_new = X - np.linalg.solve(L+N1+N2,F_X); # No Need to include lhs matrix M as it automatically cancels out.

		err = np.linalg.norm((X_new - X),2)/np.linalg.norm(X_new,2)							
		print 'Error ',err
		print 'Iteration: ', it,"\n"

		X = X_new;
		it+=1;

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))		

	#np.save(STR,X)
	return X;	