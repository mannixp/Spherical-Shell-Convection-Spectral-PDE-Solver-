#! /usr/bin/env python

#import matplotlib.pyplot as plt
import numpy as np
import sys, time
from EigenVals import *

from Projection import *
from Routines import *
import scipy 

#----------- Latex font ----
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib import rc

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
#------------------------------


## ~~~~~~~~~~~~~~~ Matrices based on evolution eqn ~~~~~~~~~~~~~~~~~~~~ ##
##~~~~~~~~~~~~~~~~ Preconditioning based on Tuckerman papaer included ~~~~~~~

# BOTTLE NECK
# Premultiplying to convert from M dX/dt = F(X) => dX/dt = M^-1 F(X)
#@profile
def Matrices_Psuedo(Pr,sigma,R,N_modes,Re_1,Re_2,Y,D): # Uses M_inv Correct

	# Extract X & Ra
	mu = Y[-1,0]; Ra = mu; X = Y[0:-1]; nr = len(R)-2;
		
	# Part a) Linear Operator
	#LL = L_0(Pr,0.0,sigma,R,N_modes) + Ra*L_1(Pr,d,R,N_modes)# Thermal Part 
	#L_Omega  = L_OM(N_modes,R,sigma,Pr,Re_2,Re_1); # Angualr Momentum Part
	#L_Cor = L_COR(N_modes,R,sigma,Pr,Re_2,Re_1); # Coriolis term vort Part
	L = L0_SP + Ra*L1_SP + Re_1*L_OM_COR; # Sparse

	F_s = (Re_1**2)*Force; #force(Pr,R,N_modes,sigma,Re_2,Re_1)
	
	# Part b) Non-Linear Terms
	N1 = N_Full(X,L.shape,N_modes,nr,MR2,MR3,D3,D2,D);
	N2 = N2_Full(X,L.shape,N_modes,nr,MR2,MR3,D3,D2,D);

	N1 = np.matmul(M_inv,N1); # Sparse
	N2 = np.matmul(M_inv,N2); # Sparse

	# Aim to make L sparse
	#Return I, N1, N2, F all multiplied by M_inv and L_inv

	Fx = np.matmul(L+N1,X) + F_s; # Sparse  L.multiply(X) + F_s + np.matmul(N1,X);
	A = L + N1 + N2; # Sparse L.todense() + N1 + N2

	return L,Fx,A;

def Evo_FF(L,FF,N1,Y): # Correct

	# a) Extract XX & g
	XX = Y[0:-1]; gg = Y[-1,0];

	# b) New Solution iterate is
	X = np.matmul(L+N1,XX) + FF;

	return X; # System state, Type Col Vector X

def Evo_FF_X(L,FF,N1,N2,Y): # Correct 
	
	# a) Linearised operator & L_inv* for preconditioning
	F_x = L+N1+N2;

	####print "Determinant ",np.linalg.det(F_x),"\n"
	#print "Bialternate Matrix ",det(F_x),"\n"

	return F_x; # Jacobian wtr X @ X_0 Type Matrix_Operator dim(X) * dim(X)

def Evo_FF_mu(Y): # Correct ??? USE SPARSE MULTIPLY !!!
	# \mu may be defined as Re_i or Ra

	# a) Extract XX
	#mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	
	# b) Multiply to set all parts independent of T to zero in XX 		
	return L1_s.multiply(Y[0:-1])
	#return np.matmul(L_Ra,Y[0:-1]); # Jacobian wrt mu @ X_0 Type Col Vector

## ~~~~~~~~~~~~~ Two_Param_funcs ~~~~~~~~~~~~~~~~~~~
def EigenVals(A): # Correct

	EIGS = np.linalg.eigvals(A); # Modify this to reduce the calculation effort ??
	idx = EIGS.argsort()[::-1];
	eigenValues = EIGS[idx];
	print "EigenVals ", eigenValues[0:5], "\n" ## Comment Out if possible
	
	return eigenValues[0:5];

def Eigs_Null_Space(A): #Correct

	# Here A must be F_X of F(X) = 0
	eigenValues, L_Vec, R_Vec = scipy.linalg.eig(A,left = True,right=True);
	
	# Re-arrange for the maximum eigenvalues & corresponding eigenvectors 
	#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	# Sort eigenvalues
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx];
	####print "Eigs_Null_Space  ",eigenValues[0:5];

	# Find first Real eigenvector
	pp = 0;
	while abs(eigenValues[pp].imag) > 1e-08:
		pp+=1;

	print "Crit Fold Lambda ",eigenValues[pp],"\n"	
	# ~~~~~ Null space ~~~~~~~~~
	## A*q_0 = 0;
	## p_0^T*A = 0; or A^T p_0 = 0

	L_Vec = L_Vec[:,idx]; R_Vec = R_Vec[:,idx];
	p_0,q_0 = L_Vec[:,pp].real,R_Vec[:,pp].real;

	#Correct Nomralization as inner product is symmetric
	q_0 = q_0/np.inner(p_0,q_0); # Normalize inner proudct q_0 = q_0/<p_0^T,q_0>
	#print "Inner Product ",np.inner(q_0,p_0)

	n = A.shape[0] + 1;
	M = np.zeros((n,n));
	
	M[0:-1,0:-1] = A[:,:];
	M[0:-1,-1] = p_0;
	M[-1,0:-1] = np.transpose(q_0);

	return M,q_0,p_0;

def Eigs_Null_Space_Fold(A): #Correct

	# Here A must be F_X of F(X) = 0
	eigenValues, L_Vec, R_Vec = scipy.linalg.eig(A,left = True,right=True);
	
	# Re-arrange for the maximum eigenvalues & corresponding eigenvectors 
	#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	# Sort eigenvalues
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx];
	print "Eigs_Null_Space  ",eigenValues[0:5];

	# Find first Real eigenvector
	pp = 0;
	while abs(eigenValues[pp].imag) > 1e-08:
		pp+=1;
	#pp = 1;
	print "Crit Fold Lambda ",eigenValues[pp],"\n"	
	# ~~~~~ Null space ~~~~~~~~~
	## A*q_0 = 0;
	## p_0^T*A = 0; or A^T p_0 = 0

	L_Vec = L_Vec[:,idx]; R_Vec = R_Vec[:,idx];
	p_0,q_0 = L_Vec[:,pp].real,R_Vec[:,pp].real;

	# Normalization
	q_0 = q_0/np.inner(q_0,q_0); # Normalize inner proudct q_0 = q_0/<q_0^T,q_0>
	p_0 = p_0/np.inner(p_0,p_0); # Normalize inner proudct q_0 = q_0/<q_0^T,q_0>
	
	return q_0,p_0;

def Eigs_Null_Space_HOPF(A): #Correct

	# Here A must be F_X of F(X) = 0
	eigenValues, L_Vec, R_Vec = scipy.linalg.eig(A,left = True,right=True);
	
	# Re-arrange for the maximum eigenvalues & corresponding eigenvectors 
	#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	# Sort eigenvalues
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx];
	print "Eigs_Null_Space  ",eigenValues[0:5];
	
	# Find first complex eigenvector

	# Find first Imaginary eigenvector
	pp = 0;
	while abs(eigenValues[pp].imag) < 1e-08:
		pp+=1;
		
	print "Crit Hopf Lambda ",eigenValues[pp],"\n"


	omega = abs(eigenValues[pp].imag)
	print "omega ",omega
	
	# ~~~~~ Null space ~~~~~~~~~
	## A*q_0 = 0;
	## p_0^T*A = 0; or A^T p_0 = 0

	L_Vec = L_Vec[:,idx]; R_Vec = R_Vec[:,idx];
	p_0,q_0 = L_Vec[:,pp],R_Vec[:,pp];

	#Correct Nomralization as inner product is symmetric
	q_0 = q_0/np.vdot(q_0,q_0); #np.vdot(p_0,q_0); # Normalize inner proudct q_0 = q_0/<p_0^T,q_0>
	p_0 = p_0/np.vdot(p_0,p_0);
	
	return q_0,p_0,omega;

#@profile
def Hess_Mat(L,A,X,w): # Uses M-inv Correct
	
	eps_r = 1e-06; ## How large or small should this be
	
	# Central Difference
	
	''' 1) Non-Linear Terms X + eps*w
	N1_p = N_Full(X+eps_r*w,M_inv.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);
	N2_p = N2_Full(X+eps_r*w,M_inv.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);

	# 2) Non-Linear Terms X - eps*w
	N1_m = N_Full(X-eps_r*w,M_inv.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);
	N2_m = N2_Full(X-eps_r*w,M_inv.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);

	AxW = (N1_p + N2_p - (N1_m + N2_m))/(2.0*eps);
	
	
	'''# Forward Difference
	# 1) Non-Linear Terms X + eps*w
	
	w_n = np.linalg.norm(w,2) # Use norm to ensure derivative scales correctly
	#N1_p = N_Full(X+eps_r*(w/w_n),L.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);
	#N2_p = N2_Full(X+eps_r*(w/w_n),L.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);
	#N_p = spsolve(MS,N1_p+N2_p);

	N1_p = N_Full(X+eps_r*(w/w_n),L.shape,N_Modes,nr,MR2,MR3,D3,D2,D);
	N2_p = N2_Full(X+eps_r*(w/w_n),L.shape,N_Modes,nr,MR2,MR3,D3,D2,D);

	N_p = np.matmul(M_inv,N1_p+N2_p);

	A_N = L + N_p;

	#print A_N.shape
	#print A.shape

	AxW = (w_n/eps_r)*(A_N-A);
	
	return AxW;

#@profile
def test_function(L,A,X,q,p): 
	
	A_uq= np.matmul(dL_Ra,q);
	g_mu = -np.inner(p,A_uq);

	# Bottle-neck
	A_xq = Hess_Mat(L,A,X,q); ## Requires large function calls - Costly 
	g_x = -np.inner(p,A_xq); # Vector g_x of Dim R^n ????
	
	return g_mu,g_x;

# ~~~~~~~~~ Evolution G, Psuedo-Jacobain G_Y ~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~ Requires both [X_0,mu_0], [X,mu] ~~~~~~~~~~~~~~~~~~~ #


#@profile
def G_Y(L,L_inv,FF,N1,N2,Y,gg_mu,gg_x): # Correct

	# 1) Unwrap Y
	mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];

	args = [L,FF,N1,N2,Y];

	# 2 ) Set Matrices  Correct
	ind = L.shape[0]; s=(ind+1,ind+1);
	GY = np.zeros(s);

	# ~~~ Line1 ~~~~ based on args !!
	#F_x = Evo_FF_X(*args); GY[0:ind,0:ind] = F_x; 
	GY[0:ind,0:ind] = np.matmul(L_inv,Evo_FF_X(*args)); ### MOST EXPENSIVE CALL
	#F_mu = Evo_FF_mu(L_Ra,L_inv,Y); 
	GY[0:ind,ind] = np.matmul(L_inv,Evo_FF_mu(dL_Ra,Y)).reshape(ind); 

	# ~~~~ Line2 ~~~~ based on args_0 !!
	GY[ind,0:ind] = gg_x;
	GY[ind,ind] = gg_mu; 
	
	return GY;

#@profile
def G(L,L_inv,FF,N1,Y,gg): # Correct
	
	# a) Extract X & mu
	XX = Y[0:-1]; mu = Y[-1,0];
		
	# b) Package arguments
	args = [L,FF,N1,Y];
		
	# c) Solving
	# F(X,mu) = 0
	X_n = np.matmul(L_inv,Evo_FF(*args)) ### MOST EXPENSIVE CALL
	
	# g(X,mu) = 0, determinan condition
	g_n = gg;

	# 3) Repackage into GG shape Y
	GG = np.zeros(Y.shape);
	GG[0:-1] = X_n; GG[-1,0] = g_n;
	
	return GG;

#@profile
def Correct_Y(Pr,sigma,R,N_modes,Re_1,Re_2,Y): # Correct
	
	it = 0; err = 1.0; gg = 1.0;	
	# 1) Update Solution
	start_time = time.time()
	while (abs(gg) > 1e-08):

		# a) Previous old solution & current prediction
		args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
		L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args); # costly 15.7% # 
		
		# b) Create the M matrix for computing g(X,\mu) parts
		if it == 0:
			# If first iteration, compute eigenvectors
			A = np.matmul(M_inv,L+N1+N2); # Solving Correct
			M,q_0,p_0 = Eigs_Null_Space(A);
						
			'''SMq = np.zeros(len(q_0)); SMp = np.zeros(len(p_0));
			nr = (len(R)-2)
			for ll in xrange(N_Modes):
				ind = ll*3*nr
				if ll%2 == 0: 
					SMq[ind:ind+3*nr] = q_0[ind:ind+3*nr];
					SMp[ind:ind+3*nr] = p_0[ind:ind+3*nr];
				elif ll%2 == 1:
					SMq[ind:ind+3*nr] = -q_0[ind:ind+3*nr];
					SMp[ind:ind+3*nr] = -p_0[ind:ind+3*nr];
			
			print "Anti symmetric"
			q_A = 0.5*(q_0 - SMq)
			Plot_Eigen(q_A,R);
			print "Symmetric"
			q_S = 0.5*(q_0 + SMq)
			Plot_Eigen(q_S,R);'''

			# 1) Compute Normed p,q,g
			#Aq = 0
			B = np.zeros(M.shape[0]); B[-1] = 1.0
			qg = np.linalg.solve(M,B); q = qg[0:-1]/np.linalg.norm(qg[0:-1],2);
			
			#A^Tp = 0
			M = np.transpose(M); pg = np.linalg.solve(M,B); 
			p = pg[0:-1]/np.linalg.norm(pg[0:-1],2);
			gg = pg[-1];

		else:
			# Use p,w of the last step to as updated eigenvectors
			A = np.matmul(M_inv,L+N1+N2); # Solving Correct
			M[0:-1,0:-1] = A
			M[-1,0:-1] = np.transpose(q_0);
			M[0:-1,-1] = p_0;

			# 1) Compute Normed p,q,g
			#Aq = 0
			B = np.zeros(M.shape[0]); B[-1] = 1.0
			qg = np.linalg.solve(M,B); q = qg[0:-1]/np.linalg.norm(qg[0:-1],2);
			
			#A^Tp = 0
			M = np.transpose(M); pg = np.linalg.solve(M,B); 
			p = pg[0:-1]/np.linalg.norm(pg[0:-1],2);
			gg = pg[-1];

		# c) Evaluate derivatives of g(X,\mu)
		if it%10==0:
			#EigenVals(A) # 32.8 %
			print "Test func ",gg
			print "Iter ",it
			print "Ra ",Y[-1,0],"\n"
			end_time = time.time()
			print("Elapsed time was %g seconds" % (end_time - start_time))
			start_time = time.time();
		#q_0,p_0,g known
		gg_mu,gg_x = test_function(L,A,Y[0:-1],q_0,p_0); # 20% costly
		
		# d) Update Vectors
		q_0 = q; p_0 = p;

		args1 = [L,L_inv,FF,N1,Y,gg];
		args2 = [L,L_inv,FF,N1,N2,Y,gg_mu,gg_x];

		# d) Compute New Soln
		GG = G(*args1); GG_Y = G_Y(*args2);
		Y_new = Y - np.linalg.solve(GG_Y,GG);
		
		# e) Check Error	
		err = np.linalg.norm(Y_new - Y,2)/np.linalg.norm(Y_new,2);

		it+=1;
		Y = Y_new;

	print "Total Iterations ",it,"\n"			
	Eigs= EigenVals(A)
	args1 = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
	return args1,Eigs;

# ~~~~~~~~~ Evolution G HOPF Bifurcation, Psuedo-Jacobain G_Y ~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~ Requires both [X_0,mu_0,q_0,omega_0]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#@profile ADD SPARSE
def G_Y_HB(L,Fx,A,Z,v_0,w_0): # Uses L_inv Correct

	# a) # Un-wrap vector Z = [X,v,w,Ra,omega]
	X = Z[0:DIM];
	v = Z[DIM:2*DIM];
	w = Z[2*DIM:3*DIM];
	Ra = Z[-2,0]; # Ra
	omega = Z[-1,0];

	# ~~~~~~ Jacobian Matrix ~~~~~~~~~
	H_Z = np.zeros((Z.shape[0],Z.shape[0]))
	I = np.eye(DIM);
	
	# b) ~~~~~~ Obtain All Sub - Matrices ~~~~~~
	Y = np.zeros((X.shape[0]+1,1))
	Y[0:-1] = X; Y[-1,0] = Ra;

	F_x = A; #np.matmul(L_inv,A);

	ind = L.shape[0];
	F_mu = np.matmul(L_Ra,X); #.reshape(ind); 
	#F_mu = L1_s.dot(X);
	
	A_xV = Hess_Mat(L,A,X,v)
	A_xW = Hess_Mat(L,A,X,w)

	A_muV = np.matmul(L_Ra,v)
	#A_muV = L1_s.dot(v);
	A_muW = np.matmul(L_Ra,w)
	#A_muW = L1_s.dot(w);

	# c) ~~~~~~ Build Block Matrix ~~~~~~~~~~~~~~~~
	# Line1
	H_Z[0:DIM,0:DIM] = F_x
	H_Z[0:DIM,-2] = F_mu.reshape(ind);
	
	# Line2
	H_Z[DIM:2*DIM,0:DIM] = A_xV; 
	H_Z[DIM:2*DIM,DIM:2*DIM] = A;
	H_Z[DIM:2*DIM,2*DIM:3*DIM] = omega*I;
	H_Z[DIM:2*DIM,-2] = A_muV.reshape(ind);
	H_Z[DIM:2*DIM,-1] = w[:,0];

	# Line3
	H_Z[2*DIM:3*DIM,0:DIM] = A_xW; 
	H_Z[2*DIM:3*DIM,DIM:2*DIM] = -omega*I
	H_Z[2*DIM:3*DIM,2*DIM:3*DIM] = A;
	H_Z[2*DIM:3*DIM,-2] = A_muW.reshape(ind);
	H_Z[2*DIM:3*DIM,-1] = -v[:,0]; 	
	
	# Line4
	H_Z[-2,DIM:2*DIM] = v_0; H_Z[-2,2*DIM:3*DIM] = w_0;
	
	# Line5
	H_Z[-1,DIM:2*DIM] = w_0; H_Z[-1,2*DIM:3*DIM] = -v_0;
	
	return H_Z;

#@profile
def G_HB(Fx,A,Z,v_0,w_0): # Uses L_inv Correct
	
	# a) # Un-wrap vector Z = [X,v,w,Ra,omega]
	X = Z[0:DIM];
	v = Z[DIM:2*DIM];
	w = Z[2*DIM:3*DIM];
	Ra = Z[-2,0]; # Ra
	omega = Z[-1,0];

	H = np.zeros(Z.shape);
	
	# b) Solving ~~~~~ F(X,mu) = 0 ~~~~~~~~~
	Y = np.zeros((X.shape[0]+1,1))
	Y[0:-1] = X; Y[-1,0] = Ra;
	
	#args = [L,FF,N1,Y];
	#H[0:DIM] = np.matmul(L_inv,Evo_FF(*args)) ### MOST EXPENSIVE CALL
	H[0:DIM] = Fx;
	# c.1) ~~~~ Solving A*v+omega*w ~~~~~~~~~
	H[DIM:2*DIM] = np.matmul(A,v) + omega*w;
	
	# c.2) ~~~~ Solving A*w-omega*v ~~~~~~~~~
	H[2*DIM:3*DIM] = np.matmul(A,w) - omega*v;
	
	# d) ~~~~ Norm Conditions ~~~~~~~~
	H[-2] = np.dot(v_0,v) + np.dot(w_0,w) - 1.0;
	H[-1] = np.dot(w_0,v) - np.dot(v_0,w);
	
	return H;

#@profile
def Correct_Y_HOPF(Pr,sigma,R,N_modes,Re_1,Re_2,Y,D,v_0,w_0,omega_0): # Correct
	
	it = 0; err = 1.0; LAM = 1.0;
	Z = np.zeros((3*DIM + 2,1));	
	Eigs = np.zeros(5)
	EIGS = Eigs
	while (abs(LAM) > 5e-05):

		args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y,D];
		L,Fx,A = Matrices_Psuedo(*args); # costly 15.7%

		if it == 0:
			
			'''# If first iteration, compute eigenvectors
			q,p,omega = Eigs_Null_Space_HOPF(A);
			v = q.real; w = q.imag;
			#print "Re(q)"; Plot_Eigen(v,R);
			#print "Im(q)"; Plot_Eigen(w,R);
			v_0 = v + 0.01*p.real; w_0 = w + 0.001*p.imag; #As am intial Condition
			'''
			# Package into vector Z = [X,v,w,Ra,omega]
			Z[0:DIM] = Y[0:-1];
			Z[DIM:2*DIM,0] = v_0;
			Z[2*DIM:3*DIM,0] = w_0;
			Z[-2,0] = Y[-1,0]; # Ra
			Z[-1,0] = omega_0;

		# c) Error & Updates
		if it%4==0:
			
			EIGS = EigenVals(A)
			pp = 0;
			while abs(EIGS[pp].imag) < 1e-08:
				pp+=1;
			LAM = EIGS[pp].real
			print EIGS
			print "abs(Lam_0) ",abs(LAM)
			
		#print "Error ",err
		print "Omega",Z[-1,0]
		print "Iter ",it
		print "Ra ",Z[-2,0],"\n"		

		# d) Compute New Soln
		args1 = [Fx,A,Z,v_0,w_0];
		args2 = [L,Fx,A,Z,v_0,w_0];

		H = G_HB(*args1); H_Z = G_Y_HB(*args2);
		Z_new = Z - np.linalg.solve(H_Z,H);
		
		# e) Check Error	
		err = np.linalg.norm(Z_new - Z,2)/np.linalg.norm(Z_new,2);

		it+=1;
		v_0 = Z[DIM:2*DIM,0]; w_0 = Z[2*DIM:3*DIM,0];
		Z = Z_new;
		Y[0:-1] = Z[0:DIM];	Y[-1,0] = Z[-2,0];

	print "Total Iterations ",it,"\n"			
	Eigs= EIGS; #EigenVals(A)
	Y[0:-1] = Z[0:DIM]; Y[-1,0] = Z[-2,0];
	args1 = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
	return args1,Eigs,v_0,w_0,omega_0;

# ~~~~~~~~~ Evolution G Fold Bifurcation, Psuedo-Jacobain G_Y ~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~ Requires both [X_0,mu_0,q_0]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#@profile
def G_Y_F(L,Fx,A,Z,q_0):
	# a) # Un-wrap vector Z = [X,q,Ra]
	X = Z[0:DIM];
	q = Z[DIM:2*DIM];
	Ra = Z[-1,0]; # Ra

	# ~~~~~~ Jacobian Matrix ~~~~~~~~~
	H_Z = np.zeros((Z.shape[0],Z.shape[0]))
	
	# b) ~~~~~~ Obtain All Sub - Matrices ~~~~~~
	Y = np.zeros((X.shape[0]+1,1))
	Y[0:-1] = X; Y[-1,0] = Ra;

	#F_x = np.matmul(L_inv,A); # Expensive, could remove L-inv here

	ind = L.shape[0];
	F_mu = np.matmul(L_Ra,X); #.reshape(ind); # Expensive, can avoid L_inv
	#F_mu = L1_s.dot(X); #
	
	A_xq = Hess_Mat(L,A,X,q)
	A_muq = np.matmul(L_Ra,q); #.reshape(ind);
	#A_muq = L1_s.dot(q);

	# c) ~~~~~~ Build Block Matrix ~~~~~~~~~~~~~~~~
	# Line1
	H_Z[0:DIM,0:DIM] = A;
	H_Z[0:DIM,-1] = F_mu.reshape(ind); 
	
	# Line2
	H_Z[DIM:2*DIM,0:DIM] = A_xq; 
	H_Z[DIM:2*DIM,DIM:2*DIM] = A;
	H_Z[DIM:2*DIM,-1] = A_muq.reshape(ind);
		
	# Line3
	H_Z[-1,DIM:2*DIM] = q_0;
	
	return H_Z;

#@profile
def G_F(Fx,A,Z,q_0):
	# a) # Un-wrap vector Z = [X,q,Ra]
	X = Z[0:DIM];
	q = Z[DIM:2*DIM];
	Ra = Z[-1,0]; # Ra

	H = np.zeros(Z.shape);
	
	# b) Solving ~~~~~ F(X,mu) = 0 ~~~~~~~~~
	Y = np.zeros((X.shape[0]+1,1))
	Y[0:-1] = X; Y[-1,0] = Ra;
	
	#args = [L,FF,N1,Y];
	#H[0:DIM] = #np.matmul(L_inv,Evo_FF(*args)) ### MOST EXPENSIVE CALL, can avoid L_inv
	H[0:DIM] = Fx;  ### MOST EXPENSIVE CALL, can avoid L_inv

	# c.1) ~~~~ Solving A*q ~~~~~~~~~
	H[DIM:2*DIM] = np.matmul(A,q);
		
	# d) ~~~~ Norm Conditions ~~~~~~~~
	H[-1] = np.dot(q_0,q) - 1.0;
	
	return H;

#@profile
def Correct_Y_Fold(Pr,sigma,R,N_modes,Re_1,Re_2,Y,D,q_0): # Correct
	
	it = 0; err = 1.0; LAM = 1.0
	Z = np.zeros((2*DIM + 1,1));
	Eigs = np.zeros(5)
	EIGS = Eigs
	#while (abs(err) > 5e-06):
	while (abs(LAM) > 1e-07):	
		args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y,D];
		L,Fx,A = Matrices_Psuedo(*args); # costly 15.7%

		if it == 0:
			''' # If first iteration, compute eigenvectors
			q,p = Eigs_Null_Space_Fold(A);		
			#print "Re(q)"; Plot_Eigen(q,R);
			q_0 = q + 0.01*p.real; #As am intial Condition '''

			# Package into vector Z = [X,v,w,Ra,omega]
			Z[0:DIM] = Y[0:-1];
			Z[DIM:2*DIM,0] = q_0;
			Z[-1,0] = Y[-1,0]; #Ra

		# c) Error & Updates; Aim to make this calculation once only;
		if it%4==0:
			
			EIGS = EigenVals(A);
			#scipy.sparse.linalg.eigs(A,5)
			pp = 0;
			while abs(EIGS[pp].imag) > 1e-08:
				pp+=1;
			LAM = EIGS[pp].real
			
			#LAM = np.linalg.det(A);
			print "Lambda_0 ",LAM
			print "Error ",err
			print "Iter ",it
			print "Ra ",Z[-1,0],"\n"
			'''
			print "Error ",err
			#print "Omega",Z[-1,0]
			print "Iter ",it
			print "Ra ",Z[-1,0],"\n"	
			'''
		# d) Compute New Soln
		args1 = [Fx,A,Z,q_0]; args2 = [L,Fx,A,Z,q_0];

		H = G_F(*args1); H_Z = G_Y_F(*args2);
		Z_new = Z - np.linalg.solve(H_Z,H);
		
		# e) Check Error	
		err = np.linalg.norm(Z_new - Z,2)/np.linalg.norm(Z_new,2);

		it+=1;
		q_0 = Z[DIM:2*DIM,0]; # Update Null-Space
		Z = Z_new;
		Y[0:-1] = Z[0:DIM];	Y[-1,0] = Z[-1,0]; # Update Solution X & Ra

	print "Total Iterations ",it,"\n"
	print "Ra ",Z[-1,0],"\n"				
	Eigs= EIGS; #EigenVals(A)
	#Eigs= EigenVals(A)
	Y[0:-1] = Z[0:DIM]; Y[-1,0] = Z[-1,0];
	args1 = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
	return args1,Eigs,q_0;

## Steps in Re_1, and Calcs Ra on bifurcation curve
#@profile
def Two_Param_Branch(Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,D,q1):

	Re_Vals = np.arange(1.85,1.7,-0.01);
	#Re_Vals = np.arange(3.71,4.1,0.01);
	Re_V_Ra = np.zeros( (2,len(Re_Vals)) );
	
	# 1) Arrays of steady states and Ra
	YvRa = np.zeros((len(Re_Vals),Y_0.shape[0],1)); 
	LambdavRa = np.zeros((len(Re_Vals),5),dtype=complex);
	QvRa = np.zeros((len(Re_Vals),Y_0.shape[0]-1,2)); # Going to be 1 less as Ra is removed
	Y = Y_0;
	for ii in xrange(len(Re_Vals)):

		Re_1 = Re_Vals[ii]; # Set Reynolds
		print "#~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~#"
		print "Reynolds Number ",Re_1,"\n"

		# a.1) Estimate Re or Ra
		if ii < 3:
			# Use linear interp
			Y_0[-1,0] = Y_0[-1,0]; # Linear interp new Ra
			#Y_0[0:-1] = Y_0[0:-1] + ; # Linear interp new X
		else:
			# Use Polyfit & Re_1 to extrap using Ra new
			f = interp1d(Re_V_Ra[0,ii-3:ii],Re_V_Ra[1,ii-3:ii],kind='linear',fill_value="extrapolate"); 
			Y_0[-1,0] = f(Re_1); print "Extrap Ra :",f(Re_1)

			#print "Re_1 :", Re_V_Ra[0,ii-3:ii]; print "Ra :",Re_V_Ra[1,ii-3:ii],"\n";

		# Solving this would cause the solution to fall of the fold	
		#Y[0:-1] = Netwon_It(Pr,Y_0[-1,0],d,R,N_Modes,Re_1,Re_2,Y_0[0:-1]); 
		'''
		# ~~~~~~~~~~~~~~~~~~~~~~ FOLD ~~~~~~~~~~~~~~~~~~~~
		# ~~~~~~~~~~ # ~~~~~~~~~ # ~~~~~~~~~ # ~~~~~~~~~~~~
		
		if ii == 0:
			# Pass null vector	
			args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y,D];
			L,Fx,A = Matrices_Psuedo(*args); # costly 15.7%
			# If first iteration, compute eigenvectors
			q,p = Eigs_Null_Space_Fold(A);	# Could be faster as calculates all the vectors
			#print "Re(q)"; Plot_Eigen(q,R);
			q_0 = q + 0.01*p.real; #As am intial Condition
			
			#q_0 = q1;
		else:
			q_0 = q;	

		# Correct
		args = [Pr,d,R,N_Modes,Re_1,Re_2,Y_0,D,q_0];
		args1,Eigs,q = Correct_Y_Fold(*args);
		Y = args1[-1];

		# Append Results
		Re_V_Ra[0,ii] = Re_1;
		Re_V_Ra[1,ii] = Y[-1,0]; # Ra
		YvRa[ii,:,:] = Y;
		QvRa[ii,:,0] = q; 
		#QvRa[ii,:,1] = w; 
		LambdavRa[ii,:] = Eigs[:]
		'''

		# ~~~~~~~~~~~~~~~~~~~~~~~ HOPF ~~~~~~~~~~~~~~~~~~~~
		# ~~~~~~~~~~ # ~~~~~~~~~ # ~~~~~~~~~ # ~~~~~~~~~~~~
		if ii == 0:
			# Pass null vector	
			args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y,D];
			L,Fx,A = Matrices_Psuedo(*args); # costly 15.7%
			# If first iteration, compute eigenvectors
			q,p,omega_0 = Eigs_Null_Space_HOPF(A);
			v = q.real; w = q.imag;	
			v_0 = v + 0.01*p.real; w_0 = w + 0.001*p.imag; #As am intial Condition
			
			#v_0 = v1; w_0 = w1; omega_0 = omega1; #As am intial Condition
		else:
			v_0 = v; w_0 = w; omega_0 = omega;

		args = [Pr,d,R,N_Modes,Re_1,Re_2,Y_0,D,v_0,w_0,omega_0];
		args1,Eigs,v,w,omega = Correct_Y_HOPF(*args);
		Y = args1[-1];

		# c) Append Results
		Re_V_Ra[0,ii] = Re_1;
		Re_V_Ra[1,ii] = Y[-1,0]; # Ra
		YvRa[ii,:,:] = Y;
		QvRa[ii,:,0] = v; 
		QvRa[ii,:,1] = w; 
		LambdavRa[ii,:] = Eigs[:]
		
		# ~~~~~~~~~~ # ~~~~~~~~~ # ~~~~~~~~~ # ~~~~~~~~~~~~
		

		# e) Update
		Y_0 = Y;
		#eps_curr = (Y[-1,0]-Ra_c)/Ra_c;
		#print "Epsilon current ",eps_curr

		# Save Files Here Also	
		np.save(STR1,YvRa);
		np.save(STR2,LambdavRa);
		np.save(STR3,Re_V_Ra);
		np.save(STR4,QvRa);	

	return YvRa,LambdavRa,Re_V_Ra;

def Nu(X):
	npr = len(R)-2
	dr = R[1]-R[0]; A_T = -(1.0+d)/d

	psi = np.zeros(N_Modes*npr)

	for k in xrange(N_Modes):
		ind = k*3*(len(R)-2)
		nr = len(R)-2;
		psi[k*nr:(k+1)*nr] = X[ind:ind+nr,0].reshape(nr);

	return np.linalg.norm(psi,2);

def Norm_v_Ra(Y):

	N = len(Y)
	X = np.zeros(N); Ra = np.zeros(N);
	npr = len(R)-2
	TT = np.zeros(npr)
	dr = R[0]-R[1]; A_T = -(1.0+d)/d
	for ii in xrange(N):
		X[ii] = Nu(Y[ii][0:-1])
		Ra[ii] = (Y[ii][-1,0] -Ra_c)/Ra_c
		# Perhaps use Nusselt -1
		#TT = Y[ii][npr:2*npr];
		#X[ii] = (3.0*TT[0]-1.5*TT[1]+(1.0/3.0)*TT[2])/(dr*A_T);

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$Ra$',fontsize=20);
	plt.ylabel(r'$||\psi||$',fontsize=20);
	plt.plot(Ra,X,'k-',linewidth=1.5);
	plt.plot(Ra,X,'y*',linewidth=0.5);
	plt.show()

def Lam_v_Ra(LAM,Y):

	N = len(Y);	Ra = np.zeros(N);
	LAM_Re = np.zeros((5,N)); LAM_IM = np.zeros((5,N));
	for ii in xrange(N):
		LAM_Re[:,ii] = LAM[ii][:].real;
		LAM_IM[:,ii] = LAM[ii][:].imag
		Ra[ii] = (Y[ii][-1,0] -Ra_c)/Ra_c

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$\epsilon$',fontsize=20);
	plt.ylabel(r'$\Re{\lambda}$',fontsize=20);
	for ii in xrange(5):
		plt.plot(Ra,LAM_Re[ii,:],'r.',linewidth=1.5);
	plt.show()	

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$\epsilon$',fontsize=20);
	plt.ylabel(r'$\Im{\lambda}$',fontsize=20);
	for ii in xrange(5):
		plt.plot(Ra,LAM_IM[ii,:],'b.',linewidth=1.5);
	plt.show()

from scipy.interpolate import interp1d
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

# Parameters to pass
l = 1;
d = 3.0;
Ra_c = 370.0;
#Ra = Ra_c*(1+0.1)
Re_1, Re_2 = 0.0,0.0;
Pr = 10.0;

Nr = 25; N_Modes = 20+1;

# Define all subsequent radial dimensions from R !!!
D,R = cheb_radial(Nr,d); nr = len(R[1:-1])
#print "D ",D.shape
print "R ",R,"\n" #.shape

DIM = 3*nr*N_Modes;
Y_0 = np.zeros((DIM+1,1))
X = np.zeros((DIM,1))
Q1 = np.zeros(DIM)
print "Dim ",DIM

# Load ICS
#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Y = np.load("1C_SN.npy"); Ra = Y[-1,0]; #Nr = 20, N_Modes = 20+1;
#Y = np.load("1C_HB.npy"); Ra = Y[-1,0]; #Nr = 20, N_Modes = 20+1;

#Y = np.load("URP_PF.npy"); Ra = Y[-1,0]; #Nr = 20, N_Modes = 40+1;
#Y = np.load("URP_HB.npy"); Ra = Y[-1,0]; #Nr = 20, N_Modes = 40+1;


#Y = np.load("URM_SN_Re5.npy"); Ra = Y[-1,0]; #Nr = 20, N_Modes = 40+1;
#Y = np.load("URM_CUSP_Re5.npy"); Ra = Y[-1,0]; #Nr = 20, N_Modes = 40+1;

#Y = np.load("Y_HB_Re40_F.npy"); Ra = Y[-1,0]; #Nr = 45, N_Modes = 40+1; 
#Re_1 = 40

#Y = np.load("1Cell2_Pr10.npy"); Ra = Y[-1,0]; #Nr = 45, N_Modes = 40+1; 
#Y = np.load("1Cell_SN_Pr10.npy"); Ra = Y[-1,0]; #Nr = 25, N_Modes = 30+1; 

Y_0 = np.load("Y_HB_LOW_Re185.npy"); Ra = Y_0[-1,0]; 
#Y_0 = np.load("1C_HB_Re15_Pr10.npy"); Ra = Y_0[-1,0];# + 10; 
Re_1 = 1.85

#print Y[0:-1,0].shape
#sys.exit()
#Y = np.load("Y_HB_Re33p5_F.npy"); Ra = Y[-1,0]; Re_1 = 33.5;
#print Ra
'''
D_o,R_o = cheb_radial(45,d);
X = INTERP_SPEC(R,R_o,Y[0:-1]);
#Q1 = INTERP_SPEC(R,R_o,Q[:,0])[:,0];
Y_0[0:-1] = Netwon_It(Pr,Ra,d,Nr,N_Modes,Re_1,Re_2,X)
Y_0[-1,0] = Ra;

from Projection import *
X = Y_0[0:-1];
f = 1e-11; Nth = 300; 
theta = np.linspace(f,np.pi-f,Nth); xx = np.linspace(R[-1],R[0],100);
#PSI, T, OMEGA, T_0 = Outer_Prod(R,theta,N_Modes,X,d,Re_1,Re_2,Pr)
PSI, T, OMEGA, T_0 = SPEC_TO_GRID(R,xx,theta,N_Modes,X,d,Re_1,Re_2,Pr)
Plot_Package_CHE(xx,theta,OMEGA,PSI,T,d,Re_1,Ra,Pr)
Energy(X,N_Modes,R);
'''

# Calculate All Linear Matrices Globally
#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MR2 = np.diag(1.0/(R[1:-1]**2)); 
MR3 = np.diag(1.0/(R[1:-1]**3));
D3 = []; D2 = [];
for ii in xrange(N_Modes):
	D3.append(D3_SP(D,R,ii));
	D2.append(Stokes_D2_SP(D,R,ii));

M = M_0_SP(D,R,N_Modes); M_inv = np.linalg.inv(M);

L0 = L_0_SP(D,R,d,Pr,0.0,N_Modes); # Ra = 0.0 here!
L1 = L_1_SP(R,d,Pr,N_Modes); # To be summed and multiplied by Ra  

L_Omega = L_OM_SP(D,N_Modes,R,d,Pr,0,1.0); # Multiply by Re_1
L_Cor = L_COR_SP(D,N_Modes,R,d,Pr,0,1.0); # Multiply by Re_1

Force = force_SP(Pr,R,N_Modes,d,0,1.0); # Multiply by (Re_1)^2

# Multiply all Linear Matrices by M_inv
L0_SP = np.matmul(M_inv,L0); L1_SP = np.matmul(M_inv,L1);
L_Cor = np.matmul(M_inv,L_Cor); L_Omega = np.matmul(M_inv,L_Omega);
Force = np.matmul(M_inv,Force)

L_OM_COR = L_Omega + L_Cor;
L_Ra = L1_SP;

# ~~~~~~~~~~~~ File Names & Code to run ~~~~~~~~~~~~~~~~ #

# File names to Save
STR1 = "".join(['Y_LOW_fwd20_HB2_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);
STR2 = "".join(['Lamda_LOW_fwd20_HB2_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']); 
STR3 = "".join(['ReRa_LOW_fwd20_HB2_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);
STR4 = "".join(['Q_LOW_fwd20_HB2_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);


start_time = time.time()
args = [Pr,d,R,N_Modes,Re_1,Re_2,Y_0,D,Q1];
YvsRa,Eigs,Re_V_Ra = Two_Param_Branch(*args);
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))

#sys.exit()
# ~~~~~~~~~~~~ Plot Solutions Out ~~~~~~~~~~~~~~~~ #
YvsRa,Eigs,Re_V_Ra = np.load(STR1),np.load(STR2),np.load(STR3);

'''
Energy(YvsRa[0],N_Modes,R);
sys.exit()
'''
#Norm_v_Ra(YvsRa); Lam_v_Ra(Eigs,YvsRa)
print Re_V_Ra
#sys.exit()
#np.save("1C_HB_Re15_Pr10.npy",YvsRa[7])
#print YvsRa[3][-1,0]

plt.xlabel(r'$Re_1$')
plt.ylabel(r'$Ra$')
plt.plot(Re_V_Ra[0,:],Re_V_Ra[1,:],'b.')
plt.show()