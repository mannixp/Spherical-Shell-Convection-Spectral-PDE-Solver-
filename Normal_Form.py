#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys, time
from EigenVals_cy import *

#from Projection import *
from Routines import *
import scipy 

#----------- Latex font ----
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#------------------------------


# ~~~~~~~~~~ # Differentiation Matrices ~~~~~~~~~~~~~~~~~~~~~~ #
def d_RaG(l,R): #Correct

	# Returns dL_{l}/dRa
	
	nr = len(R)-2;
	Z0 = np.zeros((nr,nr));
	RaG = np.diag(1.0/(R[1:-1])**2); #Bouyancy_Ra(l,R)

	if l == 0:
		Ll_0 = np.bmat([[Z0,Z0,Z0],[Z0,Z0,Z0],[Z0,Z0,Z0]]);
	else:
		Ll_0 = np.bmat([[Z0,RaG,Z0],[Z0,Z0,Z0],[Z0,Z0,Z0]]);

	return Ll_0;	

def FF_mu():# Correct

	# Returns dL/dRa
	nr = 3*(len(R)-2); Z0 = np.zeros((nr,nr));

	# Define the full FF_mu Matrix
	L_Ra = [];
	for l in xrange(N_Modes):
		L_r= [];
		for m in xrange(N_Modes):
			if m == l: # Careful this is interpreted correctly
				L_r.append(Pr*d_RaG(l,R));
			else:
				L_r.append(Z0);
		L_Ra.append(L_r)
	
	L_Ra = np.bmat(L_Ra)

	return L_Ra;
## ~~~~~~~~~~~~~~~ Matrices based on evolution eqn ~~~~~~~~~~~~~~~~~~~~ ##
##~~~~~~~~~~~~~~~~ Preconditioning based on Tuckerman papaer included ~~~~~~~

#@profile # BOTTLE NECK
# Premultiplying to convert from M dX/dt = F(X) => dX/dt = M^-1 F(X)
def Matrices_Psuedo(Pr,sigma,R,N_modes,Re_1,Re_2,Y): # Correct

	# Extract X & Ra
	mu = Y[-1,0]; Ra = mu; X = Y[0:-1];
		
	# Part a) Linear Operator
	LL = L_0(Pr,Ra,sigma,R,N_modes); # Thermal Part 
	L_Omega  = L_OM(N_modes,R,sigma,Pr,Re_2,Re_1); # Angualr Momentum Part
	L = LL + L_Omega; # Their sum
	FF = force(Pr,R,N_modes,sigma,Re_2,Re_1)

	# Part b) Non-Linear Terms
	N1 = N_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);
	N2 = N2_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);


	# c) Multiply everything by M_inv
	#L = np.matmul(M_inv,L);
	#FF = np.matmul(M_inv,FF);
	#N1,N2 = np.matmul(M_inv,N1),np.matmul(M_inv,N2);

	L_inv = np.linalg.inv(L)# Currently unused

	return L,L_inv,FF,N1,N2;

def Evo_FF(L,FF,N1,Y): # Correct

	# a) Extract XX & g
	XX = Y[0:-1]; gg = Y[-1,0];

	# b) New Solution iterate is
	X = np.matmul(L+N1,XX) + FF;

	return np.matmul(M_inv,X); # System state, Type Col Vector X

def Evo_FF_X(L,FF,N1,N2,Y): # Correct 
	
	# a) Linearised operator & L_inv* for preconditioning
	F_x = L+N1+N2;

	####print "Determinant ",np.linalg.det(F_x),"\n"
	#print "Bialternate Matrix ",det(F_x),"\n"

	return np.matmul(M_inv,F_x); # Jacobian wtr X @ X_0 Type Matrix_Operator dim(X) * dim(X)

def Evo_FF_mu(dL_Ra,Y): # Correct ???
	# \mu may be defined as Re_i or Ra

	# a) Extract XX
	#mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	
	# b) Multiply to set all parts independent of T to zero in XX 		
	
	return np.matmul(dL_Ra,Y[0:-1]); # Jacobian wrt mu @ X_0 Type Col Vector

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

## ~~~~~~~~~~~~~~~~~~~~~~~~~ ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## ~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def Vec_Eigs_Null_Space_Fold(A): #Correct

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

	# Normalization
	q_0 = q_0/np.inner(p_0,q_0); # Normalize inner proudct q_0 = q_0/<p_0^T,q_0>
	
	return q_0,p_0;

def Vec_Eigs_Null_Space_HOPF(A): #Correct

	# Here A must be F_X of F(X) = 0
	eigenValues, L_Vec, R_Vec = scipy.linalg.eig(A,left = True,right=True);
	
	# Re-arrange for the maximum eigenvalues & corresponding eigenvectors 
	#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	# Sort eigenvalues
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx];
	print "Eigs_Null_Space  ",eigenValues[0:5];
	
	# Find first complex eigenvector
	pp = 0; 
	while abs(eigenValues[pp].imag) < 1e-08:
		pp+=1;
		
	print "Crit Hopf Lambda ",eigenValues[pp],"\n"


	omega = abs(eigenValues[pp].imag)
	print "omega ",omega
	
	# ~~~~~ Null space ~~~~~~~~~
	## A*q_1 = 0;
	## A^T q_1 = 0  #Adjoint

	L_Vec = L_Vec[:,idx]; R_Vec = R_Vec[:,idx];
	p_1,q_1 = L_Vec[:,pp],R_Vec[:,pp];
	#pp = 1; pc_1,qc_1 = L_Vec[:,pp],R_Vec[:,pp];

	# ~~~~~~ Normalization ~~~~~~
	q_1 = q_1/np.vdot(p_1,q_1); #np.vdot(p_1,q_1); # Normalize inner proudct q_1 = q_1/<p_1^T,q_1>

	return q_1,p_1;

## ~~~~~~~~~~~~~~~~~~~~~~~~~ ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## ~~~~~~~~~~~~~~~~~~~~~~~~~ ##

#@profile
def Hess_Mat(L,A,X,w):
	
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
	N1_p = N_Full(X+eps_r*(w/w_n),L.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);
	N2_p = N2_Full(X+eps_r*(w/w_n),L.shape,N_Modes,nr,dr,MR2,MR3,D3,D2,D_B);

	A_N = np.matmul(M_inv,L+N1_p+N2_p);

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
		L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args); # costly 15.7%
		
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

#@profile
def G_Y_HB(L,L_inv,FF,N1,N2,A,Z,v_0,w_0): # Correct

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

	F_x = np.matmul(L_inv,A);

	ind = L.shape[0];
	F_mu = np.matmul(L_inv,Evo_FF_mu(dL_Ra,Y)).reshape(ind); 

	A_xV = Hess_Mat(L,A,X,v)
	A_xW = Hess_Mat(L,A,X,w)

	A_muV = np.matmul(dL_Ra,v)
	A_muW = np.matmul(dL_Ra,w)


	# c) ~~~~~~ Build Block Matrix ~~~~~~~~~~~~~~~~
	# Line1
	H_Z[0:DIM,0:DIM] = F_x
	H_Z[0:DIM,-2] = F_mu;
	
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
def G_HB(L,L_inv,FF,N1,A,Z,v_0,w_0): # Correct
	
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
	
	args = [L,FF,N1,Y];
	H[0:DIM] = np.matmul(L_inv,Evo_FF(*args)) ### MOST EXPENSIVE CALL

	# c.1) ~~~~ Solving A*v+omega*w ~~~~~~~~~
	H[DIM:2*DIM] = np.matmul(A,v) + omega*w;
	
	# c.2) ~~~~ Solving A*w-omega*v ~~~~~~~~~
	H[2*DIM:3*DIM] = np.matmul(A,w) - omega*v;
	
	# d) ~~~~ Norm Conditions ~~~~~~~~
	H[-2] = np.dot(v_0,v) + np.dot(w_0,w) - 1.0;
	H[-1] = np.dot(w_0,v) - np.dot(v_0,w);
	
	return H;

def Correct_Y_HOPF(Pr,sigma,R,N_modes,Re_1,Re_2,Y): # Correct
	
	it = 0; err = 1.0;
	Z = np.zeros((3*DIM + 2,1));	

	# b) Create the M matrix for computing g(X,\mu) parts

	while (err > 1e-08):

		args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
		L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args); # costly 15.7%
		A = np.matmul(M_inv,L+N1+N2);

		if it == 0:
			# If first iteration, compute eigenvectors
			q,p,omega = Eigs_Null_Space_HOPF(A);
			v = q.real; w = q.imag;
			
			#print "Re(q)"; Plot_Eigen(v,R);
			#print "Im(q)"; Plot_Eigen(w,R);
			
			v_0 = v + 0.01*p.real; w_0 = w + 0.001*p.imag; #As am intial Condition

			# Package into vector Z = [X,v,w,Ra,omega]
			Z[0:DIM] = Y[0:-1];
			Z[DIM:2*DIM,0] = v;
			Z[2*DIM:3*DIM,0] = w;
			Z[-2,0] = Y[-1,0]; # Ra
			Z[-1,0] = omega;

		# c) Error & Updates
		if it%2==0:
			#EigenVals(A) # 32.8 %
			print "Error ",err
			print "Iter ",it
			print "Ra ",Z[-2,0],"\n"		

		# d) Compute New Soln
		args1 = [L,L_inv,FF,N1,A,Z,v_0,w_0];
		args2 = [L,L_inv,FF,N1,N2,A,Z,v_0,w_0];

		H = G_HB(*args1); H_Z = G_Y_HB(*args2);
		Z_new = Z - np.linalg.solve(H_Z,H);
		
		# e) Check Error	
		err = np.linalg.norm(Z_new - Z,2)/np.linalg.norm(Z_new,2);

		it+=1;
		v_0 = Z[DIM:2*DIM,0]; w_0 = Z[2*DIM:3*DIM,0];
		Z = Z_new;
		Y[0:-1] = Z[0:DIM];	Y[-1,0] = Z[-2,0];

	print "Total Iterations ",it,"\n"			
	#Eigs= EigenVals(A)
	#Y[0:-1] = Z[0:DIM]; Y[-1,0] = Z[-2,0];
	#args1 = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
	#return args1,Eigs;
	return Y,v_0,w_0;

# ~~~~~~~~~ Evolution G Fold Bifurcation, Psuedo-Jacobain G_Y ~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~ Requires both [X_0,mu_0,q_0,omega_0]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#@profile
def G_Y_F(L,L_inv,FF,N1,N2,A,Z,q_0): # Correct

	# a) # Un-wrap vector Z = [X,q,Ra]
	X = Z[0:DIM];
	q = Z[DIM:2*DIM];
	Ra = Z[-1,0]; # Ra

	# ~~~~~~ Jacobian Matrix ~~~~~~~~~
	H_Z = np.zeros((Z.shape[0],Z.shape[0]))
	
	# b) ~~~~~~ Obtain All Sub - Matrices ~~~~~~
	Y = np.zeros((X.shape[0]+1,1))
	Y[0:-1] = X; Y[-1,0] = Ra;

	F_x = np.matmul(L_inv,A);

	ind = L.shape[0];
	F_mu = np.matmul(L_inv,Evo_FF_mu(dL_Ra,Y)).reshape(ind); 

	A_xq = Hess_Mat(L,A,X,q)
	
	A_muq = np.matmul(dL_Ra,q).reshape(ind); 

	# c) ~~~~~~ Build Block Matrix ~~~~~~~~~~~~~~~~
	# Line1
	H_Z[0:DIM,0:DIM] = F_x
	H_Z[0:DIM,-1] = F_mu;
	
	# Line2
	H_Z[DIM:2*DIM,0:DIM] = A_xq; 
	H_Z[DIM:2*DIM,DIM:2*DIM] = A;
	H_Z[DIM:2*DIM,-1] = A_muq;
		
	# Line3
	H_Z[-1,DIM:2*DIM] = q_0;
	
	return H_Z;

#@profile
def G_F(L,L_inv,FF,N1,A,Z,q_0): # Correct
	
	# a) # Un-wrap vector Z = [X,q,Ra]
	X = Z[0:DIM];
	q = Z[DIM:2*DIM];
	Ra = Z[-1,0]; # Ra

	H = np.zeros(Z.shape);
	
	# b) Solving ~~~~~ F(X,mu) = 0 ~~~~~~~~~
	Y = np.zeros((X.shape[0]+1,1))
	Y[0:-1] = X; Y[-1,0] = Ra;
	
	args = [L,FF,N1,Y];
	H[0:DIM] = np.matmul(L_inv,Evo_FF(*args)) ### MOST EXPENSIVE CALL

	# c.1) ~~~~ Solving A*q ~~~~~~~~~
	H[DIM:2*DIM] = np.matmul(A,q);
		
	# d) ~~~~ Norm Conditions ~~~~~~~~
	H[-1] = np.dot(q_0,q) - 1.0;
	
	return H;

def Correct_Y_Fold(Pr,sigma,R,N_modes,Re_1,Re_2,Y): # Correct
	
	it = 0; err = 1.0; LAM = 1.0
	Z = np.zeros((2*DIM + 1,1));

	while (abs(LAM) > 1e-10):

		args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
		L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args); # costly 15.7%
		A = np.matmul(M_inv,L+N1+N2);

		if it == 0:
			# If first iteration, compute eigenvectors
			q,p = Eigs_Null_Space_Fold(A);		
			#print "Re(q)"; Plot_Eigen(q,R);
			
			q_0 = q + 0.01*p.real; #As am intial Condition

			# Package into vector Z = [X,v,w,Ra,omega]
			Z[0:DIM] = Y[0:-1];
			Z[DIM:2*DIM,0] = q;
			Z[-1,0] = Y[-1,0]; #Ra

		# c) Error & Updates
		if it%4==0:
			LAM = EigenVals(A)[0] # 32.8 %
			print "Lambda_0 ",LAM
			print "Error ",err
			print "Iter ",it
			print "Ra ",Z[-1,0],"\n"		

		# d) Compute New Soln
		args1 = [L,L_inv,FF,N1,A,Z,q_0];
		args2 = [L,L_inv,FF,N1,N2,A,Z,q_0];

		H = G_F(*args1); H_Z = G_Y_F(*args2);
		Z_new = Z - np.linalg.solve(H_Z,H);
		
		# e) Check Error	
		err = np.linalg.norm(Z_new - Z,2)/np.linalg.norm(Z_new,2);

		it+=1;
		q_0 = Z[DIM:2*DIM,0]; # Update Null-Space
		Z = Z_new;
		Y[0:-1] = Z[0:DIM];	Y[-1,0] = Z[-1,0]; # Update Solution X & Ra

	print "Total Iterations ",it,"\n"			
	#Eigs= EigenVals(A)
	#Y[0:-1] = Z[0:DIM]; Y[-1,0] = Z[-1,0];
	#args1 = [Pr,sigma,R,N_modes,Re_1,Re_2,Y];
	#return args1,Eigs;
	return Y,q_0
## Steps in Re_1, and Calcs Ra on bifurcation curve
#@profile
def Two_Param_Branch(Pr,sigma,R,N_modes,Re_1,Re_2,Y_0):

	Re_Vals = np.arange(3.35,-0.2,-0.1);
	Re_V_Ra = np.zeros( (2,len(Re_Vals)) );
	
	# 1) Arrays of steady states and Ra
	YvRa = []; LambdavRa = [];
	Y = Y_0
	for ii in xrange(len(Re_Vals)):

		Re_1 = Re_Vals[ii]; # Set Reynolds
		print "#~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~#"
		print "Reynolds Number ",Re_1,"\n"

		# a.1) Estimate Re or Ra
		if ii < 3:
			# Use linear interp
			Y_0[-1,0] = Y_0[-1,0]-10.0; # Linear interp new Ra
			#Y_0[0:-1] = Y_0[0:-1] + ; # Linear interp new X
		else:
			# Use Polyfit & Re_1 to extrap using Ra new
			f = interp1d(Re_V_Ra[0,ii-3:ii],Re_V_Ra[1,ii-3:ii],kind='linear',fill_value="extrapolate"); 
			Y_0[-1,0] = f(Re_1); print "Extrap Ra :",f(Re_1)

			print "Re_1 :", Re_V_Ra[0,ii-3:ii]; print "Ra :",Re_V_Ra[1,ii-3:ii],"\n";

		#Y[0:-1] = Netwon_It(Pr,Y_0[-1,0],d,R,N_Modes,Re_1,Re_2,Y_0[0:-1]);

		# b) Correct
		args = [Pr,d,R,N_Modes,Re_1,Re_2,Y];
		#args1,Eigs = Correct_Y(*args);
		#args1,Eigs = Correct_Y_HOPF(*args);
		args1,Eigs = Correct_Y_Fold(*args);
		Y = args1[-1];

		# c) Append Results
		Re_V_Ra[0,ii] = Re_1;
		Re_V_Ra[1,ii] = Y[-1,0]; # Ra
		YvRa.append(Y); 
		LambdavRa.append(Eigs)

		# e) Update
		Y_0 = Y;
		#eps_curr = (Y[-1,0]-Ra_c)/Ra_c;
		#print "Epsilon current ",eps_curr

		# Save Files Here Also	
		np.save(STR1,YvRa);
		np.save(STR2,LambdavRa);
		np.save(STR3,Re_V_Ra);	

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

def Spectrum(X):
	plt.figure(figsize=(8, 6))
	E = np.zeros(N_Modes);
	for l in xrange(N_Modes):
		if l == 0:
			E[l] = np.linalg.norm(X[(len(R)-2):2*(len(R)-2),0],2);
		elif l == 1:
			E[l] = np.linalg.norm(X[3*(len(R)-2):5*(len(R)-2),0],2);
		else:
			ind = 5*(len(R)-2) + (l-2)*3*(len(R)-2);
			E[l] = np.linalg.norm(X[ind:ind+3*(len(R)-2),0],2);
	print "argmax(E) ",np.max(E),"\n"
	#E = np.log10(E[:]);
	#print E
	#E[np.isneginf(E)] = 0;
	#print E
	#print np.arange(0,N_Modes,1)[:]
	for l in xrange(N_Modes):
		#if l%2 ==0:
		plt.semilogy(l,E[l], 'b*')
	plt.xlabel(r'Mode number $\ell$', fontsize=16)
	plt.ylabel(r'$log10( ||X_{\ell}|| )$', fontsize=16)

	plt.xticks(np.arange(0,N_Modes,1))
	#plt.xlim([0,20])
	plt.show()

from scipy.interpolate import interp1d

# Parameters to pass
'''l = 3;
d = 1.8;
Ra_c = 1030.0;
Ra = Ra_c*(1+0.1)
Re_1, Re_2 = 1.0,0.0;
Pr = 10.0;

Nr = 40;
R = np.linspace(1.0,1.0+d,Nr);
N_Modes = 8+1;

ij = 15; # Close to Fold-Hopf Point

'''
l = 1;
d = 3.0;
Ra_c = 370.0;
#Ra = Ra_c*(1+0.1)
Re_1, Re_2 = 20.0,0.0;
Pr = 10.0;

Nr = 60;
R = np.linspace(1.0,1.0+d,Nr);
N_Modes = 40+1;

#ij = 49; # Close to Fold-Hopf Point for PF
ij = 0; # close to Hopf Point for HB
#ij = -1; # Last point on Saddle Node

Stringy = ['/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/Pseudo/CoDim2_l1/']
os.chdir("".join(Stringy))

# Calculate all the associated Linear matrices once, globally
## Sparsify these

#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nr = len(R)-2; dr = R[1] - R[0];
MR2 = np.diag(1.0/(R[1:-1]**2)); 
MR3 = np.diag(1.0/(R[1:-1]**3));
D3 = []; D2 = []; D_B = D_b(R);
for ii in xrange(N_Modes):
	D3.append(D3_l(ii,R));
	D2.append(Stokes_D2(R,ii));
M_rhs = M_0(R,N_Modes);
M_inv = np.linalg.inv(M_rhs);
dL_Ra = FF_mu();
dL_Ra = np.matmul(M_inv,dL_Ra);

DIM = 3*nr*N_Modes; # Correct

# ~~~~~~~~~~~~ File Names & Code to run ~~~~~~~~~~~~~~~~ #

'''# File names to Load
STR1 = "".join(['Y_fwd_SN_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);
STR2 = "".join(['Lamda_fwd_SN_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']); 
STR3 = "".join(['ReRa_fwd_SN_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);
'''

STR1 = "".join(['Y_fwd_HB_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']); # N_theta = 20+1
STR2 = "".join(['Lambda_fwd_HB_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']); 
STR3 = "".join(['ReRa_fwd_HB_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);

'''
STR1 = "".join(['Y_test_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']); # N_theta = 20+1
STR2 = "".join(['Lamda_test_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']); 
STR3 = "".join(['ReRa_test_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);
'''
#args = [Pr,d,R,N_Modes,Re_1,Re_2,Y_0];
#YvsRa,Eigs,Re_V_Ra = Two_Param_Branch(*args);
#sys.exit()

YvsRa,Eigs,Re_V_Ra = np.load(STR1),np.load(STR2),np.load(STR3);

# ~~~~~~~~~~~~  Interpolate onto a finer grid ~~~~~~~~~~~~~~~~ #

Y = YvsRa[ij,:,:];
Re_1 = Re_V_Ra[0,ij]; 
Ra = Re_V_Ra[1,ij]; 
#Ra = 11638.9;
Y[-1,0] = Ra;
print Y.shape
print "Re_1 ",Re_1
print "Ra ",Ra
print "Eigs ", Eigs[ij]

'''for ii in xrange(len(Eigs)):
	print "Iteration ",ii,"\n" 
	print Re_V_Ra[:,ii]
	print YvsRa[ii,-1,0]
	print Eigs[ii]
'''
#sys.exit()


Y_0 = np.zeros((DIM+1,1))
#Y = np.load("Y_1_Cell_HB_Init.npy")
X = INTERP(R,np.linspace(1,1+d,60),Y[0:-1])
Y_0[0:-1] = Netwon_It(Pr,Y[-1,0],d,R,N_Modes,Re_1,Re_2,X)
Y_0[-1,0] = Y[-1,0]
#np.save("Y_1_Cell_FH_Init.npy",Y_0); # N_modes = 20+1, Nr = 60;


Stringy = ['/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/Pseudo/']
os.chdir("".join(Stringy))


#sys.exit()
# ~~~~~~~~ Resolve for $Re(\lambda) = 0$ ~~~~~~~~~~~~~~~
'''start_time = time.time()
Y_0,q_0 = Correct_Y_Fold(Pr,d,R,N_Modes,Re_1,Re_2,Y_0); print "Ra_c Fold ",Y_0[-1,0],"\n"
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
'''
start_time = time.time()
Y,u_0,w_0 = Correct_Y_HOPF(Pr,d,R,N_Modes,Re_1,Re_2,Y_0); print "Ra_c Hopf ",Y_0[-1,0],"\n"
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

args = [Pr,d,R,N_Modes,Re_1,Re_2,Y_0];
L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args); # costly 15.7%
A = np.matmul(M_inv,L+N1+N2);

q_1,p_1 = Vec_Eigs_Null_Space_HOPF(A); # Complex Eigenvectors for s = i \omega
#q_0,p_0 = Vec_Eigs_Null_Space_Fold(A); # Real Eigenvectors for s=0


#sys.exit()
# ~~~~~~~~~~~~ Plot Solutions Out ~~~~~~~~~~~~~~~~ #

#Norm_v_Ra(YvsRa);
#Lam_v_Ra(Eigs,YvsRa)

#for ii in xrange(10):	
#	print Eigs[ii][:]
#print Re_V_Ra[:,0:20]

print "Steady Solution"
X = Y_0[0:-1]; #q_1.real
print X.shape
from Projection import *
f = 1e-11; Nth = 100;
theta = np.linspace(f,np.pi-f,Nth)
PSI, T, OMEGA, T_0 = Outer_Prod(R,theta,N_Modes,X,d,Re_1,Re_2,Pr);
Plot_Package_CHE(R,theta,OMEGA,PSI,T,d,Re_1,Ra,Pr)

Spectrum(X)

print "SN Eigenvector real"
X[:,0] = q_1; #.real

PSI, T, OMEGA, T_0 = Outer_Prod(R,theta,N_Modes,X,d,Re_1,Re_2,Pr);
Plot_Package_CHE(R,theta,OMEGA,PSI,T,d,Re_1,Ra,Pr)

Spectrum(X)

'''print "Hopf Eigenvector Imag"
X[:,0] = q_1.imag

PSI, T, OMEGA, T_0 = Outer_Prod(R,theta,N_Modes,X,d,Re_1,Re_2,Pr);
Plot_Package_CHE(R,theta,OMEGA,PSI,T,d,Re_1,Ra,Pr)

Spectrum(X)'''