#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys, time, scipy
from EigenVals import *

from Projection import *
from Routines import *

#----------- Latex font ----
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#------------------------------


## ~~~~~~~~~~~~~~~ Matrices based on evolution eqn ~~~~~~~~~~~~~~~~~~~~ ##
##~~~~~~~~~~~~~~~~ Preconditioning based on Tuckerman papaer included ~~~~~~~

#@profile # BOTTEL NECK			
def Matrices_Psuedo(Pr,sigma,R,N_modes,Re_1,Re_2,Y,D): # Correct

	# Extract X & Ra
	mu = Y[-1,0]; Ra = mu; X = Y[0:-1]; nr = len(R)-2;
		
	# Part a) Linear Operator
	#LL = L_0(Pr,0.0,sigma,R,N_modes) + Ra*L_1(Pr,d,R,N_modes)# Thermal Part 
	#L_Omega  = L_OM(N_modes,R,sigma,Pr,Re_2,Re_1); # Angualr Momentum Part
	#L_Cor = L_COR(N_modes,R,sigma,Pr,Re_2,Re_1); # Coriolis term vort Part
	L = L0 + Ra*L1 + Re_1*L_OM_COR; # Their sum

	'''
	plt.spy(L,precision=1e-12,markersize=5) # Very sparse
	#plt.spy(M_inv, precision=1e-12,markersize=5)
	plt.show()
	sys.exit()
	'''
	FF = (Re_1**2)*Force; #force(Pr,R,N_modes,sigma,Re_2,Re_1)

	L_inv = np.linalg.inv(L)
	
	# Part b) Non-Linear Terms
	N1 = N_Full(X,L.shape,N_modes,nr,MR2,MR3,D3,D2,D);
	N2 = N2_Full(X,L.shape,N_modes,nr,MR2,MR3,D3,D2,D);

	# Aim to make L sparse
	#Return I, N1, N2, F all multiplied by M_inv and L_inv

	return L,L_inv,FF,N1,N2;

def Evo_FF(L,L_inv,FF,N1,N2,Y): # Correct

	# a) Extract XX
	mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];

	# b) New Solution iterate is
	X = np.matmul(L+N1,XX) + FF;

	# c0 L_inv* for preconditioning
	return np.matmul(L_inv,X); # System state, Type Col Vector X

def Evo_FF_X(L,L_inv,FF,N1,N2,Y): # Correct 
	
	# a) Linearised operator & L_inv* for preconditioning
	F_x = np.matmul(L_inv,L+N1+N2);

	####print "Determinant ",np.linalg.det(F_x),"\n"
	#print "Bialternate Matrix ",det(F_x),"\n"

	return F_x; # Jacobian wtr X @ X_0 Type Matrix_Operator dim(X) * dim(X)

def Evo_FF_mu(L_Ra,L_inv,Y): # Correct
	# \mu may be defined as Re_i or Ra

	# a) Extract XX
	#mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	
	# b) Multiply to set all parts independent of T to zero in XX 		
	
	return np.matmul(L_inv,np.matmul(L_Ra,Y[0:-1])); # Jacobian wrt mu @ X_0 Type Col Vector


## ~~~~~~~~~~~~~~~ Matrices based on evolution eqn ~~~~~~~~~~~~~~~~~~~~ ##
#@profile
def xi(L,L_inv,FF,N1,N2,Y):
	args = [L,L_inv,FF,N1,N2,Y]
	
	Fx,F_mu = Evo_FF_X(*args),Evo_FF_mu(L_Ra,L_inv,Y);
	
	return (-1.0)*np.linalg.solve(Fx,F_mu);

def mu_dot(chi):
	
	# derivative of N wrt mu @ mu_0, Type Scalar
	return sign/np.sqrt(1.0 + delta*( pow(np.linalg.norm(chi,2),2) -1.0) );

def X_dot(chi):

	# derivative of N wrt X @ X_0, Type Col Vector
	return mu_dot(chi)*chi; 

#@profile
def D_dots(chi):

	mu_dot = sign/np.sqrt(1.0 + delta*( pow(np.linalg.norm(chi,2),2) -1.0) );

	return mu_dot, mu_dot*chi;

## ~~~~~~~~~~~~~ Eigenvalue Computations ~~~~~~~~~~~~~~~~~~~
def EigenVals(A):

	EIGS = np.linalg.eigvals(A); # Modify this to reduce the calculation effort ??
	idx = EIGS.argsort()[::-1];
	eigenValues = EIGS[idx];
	print "EigenVals ", eigenValues[0:10], "\n" ## Comment Out if possible
	
	return eigenValues[0:5];

'''
## ~~~~~~~~~~~~~ Two_Param_funcs ~~~~~~~~~~~~~~~~~~~
def Netwon_It_2Param(Pr,sigma,R,N_modes,Re_1,Re_2,Y):

	L,L_inv,FF,N1,N2 = Matrices_Psuedo(Pr,sigma,R,N_modes,Re_1,Re_2,Y)
	X = Y[0:-1];

	# 1) Determine Steady State
	it, err = 0,1.0; 
	MAX_ITERS = 10;
	while (err > (1e-10)):	

		# a) Eval matrices
		if it > 0:
			N1 = N_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);
			N2 = N2_Full(X,L.shape,N_modes,nr,dr,MR2,MR3,D3,D2,D_B);
		
		# b) Precondition terms
		F_X = np.matmul(L_inv, np.matmul(L+ N1,X) + FF); 
		DF_X = np.matmul(L_inv,L+N1+N2);

		# c) Solve for new soln
		X_new = X - np.linalg.solve(DF_X,F_X);

		# d) Compute Error	
		err = np.linalg.norm((X_new - X),2)/np.linalg.norm(X_new,2)							
		print 'Error ',err
		print 'Iteration: ', it,"\n"

		X = X_new;
		it+=1;
		if (it > MAX_ITERS):
			break

	Y[0:-1] = X;

	# 2) Determine Eigs	
	AA = np.matmul(M_inv,L+N1+N2);
	Eig1 = EigenVals(AA)[0:5];

	return Y, it, Eig1;

def Soln_Interp()
	
	for pp in xrange(len(X)):
		x = np.zeros(3); y = np.zeros(3)
		for jj in xrange(3):
			y[jj] = YvRa[-(jj+1)][pp,0];
			x[jj] = YvRa[-(jj+1)][-1,0];
		a = np.polyfit(x,y,2)
		Y[pp,0] = np.polyval(a,Y[-1,0]);
'''
# ~~~~~~~~~ Evolution G, Psuedo-Jacobain G_Y ~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~ Requires both [X_0,mu_0], [X,mu] ~~~~~~~~~~~~~~~~~~~ #

#@profile
def G_Y(L,L_inv,FF,N1,N2,Y,Y_0,dX_0,dmu_0): # Correct

	# 1) Unwrap Y
	mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1]; # Old solution for tangent

	args = [L,L_inv,FF,N1,N2,Y];

	# 2 ) Set Matrices  Correct
	ind = L.shape[0]; s=(ind+1,ind+1);
	GY = np.zeros(s);

	# ~~~ Line1 ~~~~ based on args !!
	#F_x = Evo_FF_X(*args); GY[0:ind,0:ind] = F_x; 
	GY[0:ind,0:ind] = Evo_FF_X(*args); ### MOST EXPENSIVE CALL
	#F_mu = Evo_FF_mu(L_Ra,L_inv,Y); 
	GY[0:ind,ind] = Evo_FF_mu(L_Ra,L_inv,Y).reshape(ind); 

	# ~~~~ Line2 ~~~~ based on args_0 !!
	GY[ind,0:ind] = delta*np.transpose(dX_0);
	GY[ind,ind] = (1.0-delta)*dmu_0; 
	
	return GY;

#@profile
def G(L,L_inv,FF,N1,N2,Y,Y_0,dX_0,dmu_0): # Correct
	
	# a) Extract X & mu
	mu = Y[-1,0]; Ra = mu; XX = Y[0:-1];
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1]; # Old solution for tangent
	
	# b) Package arguments
	args = [L,L_inv,FF,N1,N2,Y];
		
	# c) Solving
	
	# Solve F(X,mu) = 0
	X_n = Evo_FF(*args) ### MOST EXPENSIVE CALL
	
	# Solve for N(X,mu,s) = 0, tangent condition
	N_n = delta*np.dot(np.transpose(dX_0),XX-X_0)[0,0] + (1.0-delta)*dmu_0*(mu - mu_0) - ds;

	# 3) Repackage into GG shape Y
	GG = np.zeros(Y.shape);
	GG[0:-1] = X_n; GG[-1,0] = N_n;
	
	return GG;

#@profile
def Pred_Y(Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,D): # Correct
		
	# 1) Unwrap Y
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1]; 

	args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,D];
	L,L_inv,FF,N1_0,N2_0 = Matrices_Psuedo(*args);
	args_0 = [L,L_inv,FF,N1_0,N2_0,Y_0];

	chi = xi(*args_0);
	dmu_dt, dx_dot =  D_dots(chi); #mu_dot(chi), X_dot(chi);

	'''det = np.linalg.det(np.matmul(L_inv,L+N1_0+N2_0))
	print "Determinant ",det

	if (det*prev_det < 0) and (abs(det) < 0.1):
		print "Bifurcation Change Determinant sign \n"
		#global sign
		#sign = -1*sign;
	global prev_det
	prev_det = det;

	print "dmu_0 ",dmu_dt
	print "||xi|| ",np.linalg.norm(dx_dot/dmu_dt,2),"\n"'''

	# 2) Predict & Update new guess
	Y_p = np.zeros(Y_0.shape);
	Y_p[0:-1] = X_0 + dx_dot*ds;
	Y_p[-1,0] = mu_0 + dmu_dt*ds; 

	return Y_p, dmu_dt, dx_dot;

#@profile
# Approx 40 times faster
def Pred_Y_Eff(GG_Y,Y_0):

	# 1) Unwrap Y + Create RHS
	mu_0 = Y_0[-1,0]; Ra_0 = mu_0; X_0 = Y_0[0:-1];
	B = np.zeros(Y_0.shape); B[-1,0] = 1.0;


	# 2) Calculate & Normalize
	Y_dot = np.linalg.solve(GG_Y,B);
	
	mu = Y_dot[-1,0];  X = Y_dot[0:-1];
	Y_dot = Y_dot/np.sqrt(delta*np.linalg.norm(X) + (1-delta)*np.linalg.norm(mu) );

	# 3) Unpack & Update new guess
	Y_p = np.zeros(Y_0.shape);
	dmu_dt, dx_dot = Y_dot[-1,0], Y_dot[0:-1];
	Y_p[0:-1] = X_0 + dx_dot*ds;
	Y_p[-1,0] = mu_0 + dmu_dt*ds;

	return Y_p, dmu_dt, dx_dot;

#@profile
def Correct_Y(Pr,sigma,D,R,N_modes,Re_1,Re_2,Y_0,Y,dmu_0,dX_0,c_eig): # Correct
	
	it = 0; err = 1.0
	# 1) Update Solution
	while err > 1e-10:
	
		# a) Previous old solution & current prediction
		args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y,D];
		L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args);
		args1 = [L,L_inv,FF,N1,N2,Y,Y_0,dX_0,dmu_0];

		# b) Update new solution using solve
		GG = G(*args1); GG_Y = G_Y(*args1);
		Y_new = Y - np.linalg.solve(GG_Y,GG);

		# Force psi_0, Omega_0 to 0
		#Y_new[len(R[1:-1]):3*len(R[1:-1]),0] = np.zeros(2*len(R[1:-1]));

		err = np.linalg.norm(Y_new - Y,2)/np.linalg.norm(Y_new,2);
		print "Iter ",it
		print "Error ",err
		print "Ra ",Y[-1,0],"\n"

		it+=1;
		Y = Y_new;

		# c) If it > 5 Repredict !!!!
		if it > 5:
			it = 0;
			global ds
			ds = ds/4.0; # Smaller for better cusp following
			print "Repredicting: reduced ds ",ds,"\n"
			
			#args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0];
			#L,L_inv,FF,N1_0,N2_0 = Matrices_Psuedo(*args);
			#args_0 = [L,L_inv,FF,N1_0,N2_0,Y_0];
			
			#chi = xi(*args_0); dmu_0, dX_0 =  D_dots(chi);
			Y[0:-1] = Y_0[0:-1] + dX_0*ds;
			Y[-1,0] = Y_0[-1,0] + dmu_0*ds;


	print "Total Iterations ",it,"\n"			
	# 2) Calculate Leading Eigenvectors
	#~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~
	if c_eig%eig_freq==0:
		A = np.matmul(M_inv,L+N1+N2);	
		eigenValues = EigenVals(A); ####
	else:
		eigenValues = [];
	#~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~
	
	# 3) Step Length Control		
	if it <= 2:
		#print ds
		global ds
		if ds < 250.0:
			ds = 2.0*ds;
			print "Increased ds ",ds,"\n"
		else:
			ds = 250.0	

	args1 = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,Y];
	return args1,eigenValues,GG_Y;

#@profile
def Psuedo_Branch(Pr,sigma,D,R,N_modes,Re_1,Re_2,Y_0,Eig1):

	# 1) Arrays of steady states and Ra
	YvRa = []; LambdavRa = [];

	# Add initial Eig1 and steady state
	YvRa.append(Y_0); LambdavRa.append(Eig1)

	eps_curr = (Y_0[-1,0]-Ra_c)/Ra_c; ii = 0;
	Eig_Sign = 1.0; diff = -1.0;
	while (ii < iters): #(Eig_Sign.real > 0.0): #(ii < iters) or 

		print "#~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~#"
		print "Branch point ",ii,"\n"
		
		# a) Pred Initial guess for [X(s),mu(s)] with step ds
		if ii < 1:	
			args = [Pr,sigma,R,N_modes,Re_1,Re_2,Y_0,D];
			Y, dmu_dt, dx_dot = Pred_Y(*args);
		else:	
			Y, dmu_dt, dx_dot = Pred_Y_Eff(GG_Y,Y_0);
			
		# b) Correct
		args = [Pr,d,D,R,N_Modes,Re_1,Re_2,Y_0,Y, dmu_dt, dx_dot, ii];
		args1,Eigs,GG_Y = Correct_Y(*args);
		Y = args1[-1];

		# c) Append Results 
		YvRa.append(Y); 

		if Eigs != []:
			LambdavRa.append(Eigs)

		# e) Termination Condition
		'''
		if ii > eig_freq:
			#print LambdavRa[ii][0]
			#print LambdavRa[ii-1][0]
			Eig_Sign = LambdavRa[ii][0]*LambdavRa[ii-1][0];
			print "Eig_sign ",Eig_Sign.real
			print "Ra ",Y[-1,0]
		'''
		# e) Update
		Y_0 = Y;
		eps_curr = (Y[-1,0]-Ra_c)/Ra_c;
		#print "Epsilon current ",eps_curr
		ii+=1

		np.save(STR1,YvRa)
		np.save(STR2,LambdavRa)

	return YvRa,LambdavRa;

def Nu(X):
	npr = len(R)-2
	dr = R[1]-R[0]; A_T = -(1.0+d)/d

	psi = np.zeros(N_Modes*npr)

	for k in xrange(N_Modes):
		ind = k*3*(len(R)-2)
		nr = len(R)-2;
		psi[k*nr:(k+1)*nr] = X[ind:ind+nr,0].reshape(nr);

	return np.linalg.norm(psi,2);

from mpmath import *
from sympy import *
import scipy.special as sp
from scipy import optimize

def test_func(x, a, b):
    return a*(x**b);

def Temp_Prof(Y):

	N = len(Y); npr = len(R)-2; TT = np.zeros(len(R))

	A_T = -(1.0+d)/d

	ii = -1; #N-1;
	#Ra[ii] = Y[ii][-1,0];#(Y[ii][-1,0] -Ra_c)/Ra_c
	
	# T_0(r)
	TT[1:-1] = Y[ii][npr:2*npr,0];
	
	# 1) ~~~~~~~~ horizontally averaged temp profile ~~~~~~~~~~
	plt.figure(figsize=(12,8))
	
	# Lowest Ra
	TT[1:-1] = Y[0][npr:2*npr,0];
	xx = np.linspace(R[-1],R[0],100);
	T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT,'bo',markerfacecolor='none',markersize=5.0)
	plt.plot(xx,T,'b-',linewidth=1.5,label=r'$Ra/Ra_c=$%3.2f'%(Y[0][-1,0]/Ra_c))


	# Highest Ra
	TT = np.zeros(len(R))
	TT[1:-1] = Y[N/2][npr:2*npr,0];
	xx = np.linspace(R[-1],R[0],100);
	T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT,'ko',markerfacecolor='none',markersize=5.0) 
	plt.plot(xx,T,'k-.',linewidth=1.5,label=r'$Ra/Ra_c=$%3.2f'%(Y[N/2][-1,0]/Ra_c))


	# Highest Ra
	TT = np.zeros(len(R))
	TT[1:-1] = Y[-1][npr:2*npr,0];
	xx = np.linspace(R[-1],R[0],100);
	T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT,'ro',markerfacecolor='none',markersize=5.0) 
	plt.plot(xx,T,'r--',linewidth=1.5,label=r'$Ra/Ra_c=$%3.2f'%(Y[-1][-1,0]/Ra_c))


	plt.xlabel(r'$r$',fontsize=20);
	plt.ylabel(r'$\overline{T(r,\theta)}$',fontsize=20);
	plt.xlim([1.0,1.0+d])
	plt.legend(fontsize=14)
	
	plt.savefig("T_radial_VsR.eps", format='eps', dpi=1800)
	plt.show()


	# 2) ~~~~~~~~ Local to Poles theta = 1,89 ~~~~~~~~~~
	'''
	P = lambda x,kk: legendre(kk,cos(x))
	f = 1e-11; Nth = 300; theta = np.linspace(f,(np.pi/2.0)-f,Nth);
	xx = np.linspace(R[-1],R[0],100);
	
	print "Pole ",theta[1]*(180.0/np.pi)
	print "Equator ",theta[-1]*(180.0/np.pi)
	
	T_p = np.zeros(len(R))
	T_e = np.zeros(len(R))

	# T(r,\theta = 1)
	for i in xrange(N_Modes):
		ind = i*3*npr
		T_p[1:-1] += Y[i][ind+npr:ind+2*npr,0]*float(P(theta[1],i));
		T_e[1:-1] += Y[i][ind+npr:ind+2*npr,0]*float(P(theta[-1],i));
	
	
	plt.figure(figsize=(12,8))
	
	# Pole
	T = np.polyval(np.polyfit(R,T_p,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	T_p = T_p + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(xx,T,'k-',linewidth=1.5,label=r'$\theta \approx 0$')
	plt.plot(R,T_p,'ko',markersize=3.0)

	# Equator
	T = np.polyval(np.polyfit(R,T_e,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	T_e = T_e + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(xx,T,'b-',linewidth=1.5,label=r'$\theta \approx \pi/2$')
	plt.plot(R,T_e,'bo',markersize=3.0)


	plt.xlabel(r'$r$',fontsize=20);
	plt.ylabel(r'$T(r,\theta)$',fontsize=20);
	plt.legend()
	plt.xlim([1.0,1.0+d])
	plt.show()
	'''
	# 3) ~~~~~~~~ Polar temp profile ~~~~~~~~~~

	P = lambda x,kk: legendre(kk,cos(x))
	f = 1e-11; Nth = 100; theta = np.linspace(f,np.pi,Nth);
	xx = np.linspace(R[-1],R[0],100);
	
	T_0 = (-A_T/R[npr/2]) - (1.0/d);
	T_m1 = T_0*np.ones(len(theta))
	T_m2 = T_0*np.ones(len(theta))
	T_m3 = T_0*np.ones(len(theta))

	for i in xrange(N_Modes):
		ind = i*3*npr
		tt1 = Y[0][ind+npr+npr/2,0]
		tt2 = Y[N/2][ind+npr+npr/2,0]
		tt3 = Y[-1][ind+npr+npr/2,0]
		for jj in xrange(len(theta)):
			Pol = float(P(theta[jj],i));
			T_m1[jj] += tt1*Pol
			T_m2[jj] += tt2*Pol
			T_m3[jj] += tt3*Pol
	
	plt.figure(figsize=(12,8))
	
	# Mid point r = 1 + 0.5*d
	plt.plot(theta,T_m1,'b-',linewidth=1.5,label=r'$Ra/Ra_c=$%3.2f'%(Y[0][-1,0]/Ra_c))
	plt.plot(theta,T_m2,'k-.',linewidth=1.5,label=r'$Ra/Ra_c=$%3.2f'%(Y[N/2][-1,0]/Ra_c))
	plt.plot(theta,T_m3,'r--',linewidth=1.5,label=r'$Ra/Ra_c=$%3.2f'%(Y[-1][-1,0]/Ra_c))

	plt.xlabel(r'$\theta$',fontsize=20);
	plt.ylabel(r'$T(r = 1 + d/2,\theta)$',fontsize=20);
	plt.legend(fontsize=14)
	plt.xlim([0,np.pi])
	plt.savefig("T_polar_Vs_theta.eps", format='eps', dpi=1800)
	plt.show()


def Temp_Prof_ESP(YL):

	N = len(YL[0]); s=(3,N)

	Y0 = YL[0]; Y1 = YL[1]; Y2 = YL[2];

	N = len(Y0); npr = len(R)-2; 
	TT_0 = np.zeros(len(R)); TT_1 = np.zeros(len(R)); TT_2 = np.zeros(len(R))

	A_T = -(1.0+d)/d
	
	# T_0(r)
	#ii = 200 # Adjust this one
	TT_0[1:-1] = Y0[npr:2*npr,0];
	TT_1[1:-1] = Y1[npr:2*npr,0];
	TT_2[1:-1] = Y2[npr:2*npr,0];
	
	xx = np.linspace(R[-1],R[0],100);

	# 1) ~~~~~~~~ horizontally averaged temp profile ~~~~~~~~~~
	plt.figure(figsize=(12,8))
	
	# 1-cell
	T = np.polyval(np.polyfit(R,TT_0,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT_0 = TT_0 + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT_0,'ko',markerfacecolor='none',markersize=5.0)
	plt.plot(xx,T,'k-',linewidth=1.5,label=r'$u_r^+$ $Ra/Ra_c=$%3.2f'%(Y0[-1,0]/Ra_c))


	# ur^+
	T = np.polyval(np.polyfit(R,TT_1,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT_1 = TT_1 + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT_1,'bo',markerfacecolor='none',markersize=5.0) 
	plt.plot(xx,T,'b--',linewidth=1.5,label=r'Mixed 1-cell unstable $Ra/Ra_c=$%3.2f'%(Y1[-1,0]/Ra_c))


	# ur^-
	T = np.polyval(np.polyfit(R,TT_2,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	TT_2 = TT_2 + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(R,TT_2,'yo',markerfacecolor='none',markersize=5.0) 
	plt.plot(xx,T,'y-',linewidth=1.5,label=r'Mixed 1-cell stable $Ra/Ra_c=$%3.2f'%(Y2[-1,0]/Ra_c))


	plt.tick_params(axis="x", labelsize=25)
	plt.tick_params(axis="y", labelsize=25)

	plt.xlabel(r'Radial position $r$',fontsize=30);
	plt.ylabel(r'$\overline{T(r,\theta)}$',fontsize=30);
	plt.ylim([0,1])
	plt.xlim([1.0,1.0+d])
	plt.legend(fontsize=25)
	plt.grid()
	plt.savefig("T_radial_VsR.eps", format='eps', dpi=1800)
	plt.show()


	# 2) ~~~~~~~~ Velocity Profile ~~~~~~~~~~
	'''
	P = lambda x,kk: legendre(kk,cos(x))
	f = 1e-11; Nth = 300; theta = np.linspace(f,(np.pi/2.0)-f,Nth);
	xx = np.linspace(R[-1],R[0],100);
	
	print "Pole ",theta[1]*(180.0/np.pi)
	print "Equator ",theta[-1]*(180.0/np.pi)
	
	T_p = np.zeros(len(R))
	T_e = np.zeros(len(R))

	# T(r,\theta = 1)
	for i in xrange(N_Modes):
		ind = i*3*npr
		T_p[1:-1] += Y[i][ind+npr:ind+2*npr,0]*float(P(theta[1],i));
		T_e[1:-1] += Y[i][ind+npr:ind+2*npr,0]*float(P(theta[-1],i));
	
	
	plt.figure(figsize=(12,8))
	
	# Pole
	T = np.polyval(np.polyfit(R,T_p,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	T_p = T_p + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(xx,T,'k-',linewidth=1.5,label=r'$\theta \approx 0$')
	plt.plot(R,T_p,'ko',markersize=3.0)

	# Equator
	T = np.polyval(np.polyfit(R,T_e,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
	T_e = T_e + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
	plt.plot(xx,T,'b-',linewidth=1.5,label=r'$\theta \approx \pi/2$')
	plt.plot(R,T_e,'bo',markersize=3.0)


	plt.xlabel(r'$r$',fontsize=20);
	plt.ylabel(r'$T(r,\theta)$',fontsize=20);
	plt.legend()
	plt.xlim([1.0,1.0+d])
	plt.show()
	'''
	# 3) ~~~~~~~~ Polar temp profile ~~~~~~~~~~

	P = lambda x,kk: legendre(kk,cos(x))
	f = 1e-11; Nth = 500; theta = np.linspace(f,np.pi,Nth);
	xx = np.linspace(R[-1],R[0],100);
	
	T_0 = (-A_T/R[npr/2]) - (1.0/d);
	T_m1 = T_0*np.ones(len(theta))
	T_m2 = T_0*np.ones(len(theta))
	T_m3 = T_0*np.ones(len(theta))

	for i in xrange(N_Modes):
		ind = i*3*npr
		tt1 = Y0[ind+npr+npr/2,0]
		tt2 = Y1[ind+npr+npr/2,0]
		tt3 = Y2[ind+npr+npr/2,0]
		for jj in xrange(len(theta)):
			Pol = float(P(theta[jj],i));
			T_m1[jj] += tt1*Pol
			T_m2[jj] += tt2*Pol
			T_m3[jj] += tt3*Pol
	
	plt.figure(figsize=(12,8))
	plt.tick_params(axis="x", labelsize=25)
	plt.tick_params(axis="y", labelsize=25)
	
	# Mid point r = 1 + 0.5*d
	plt.plot(theta,T_m1,'k-',linewidth=1.5,label=r'$u_r^+$')
	plt.plot(theta,T_m2,'b--',linewidth=1.5,label=r'Mixed 1-cell unstable')
	plt.plot(theta,T_m3,'y-',linewidth=1.5,label=r'Mixed 1-cell stable')

	plt.xlabel(r'Latitude $\theta$',fontsize=30);
	plt.ylabel(r'$T(r = 1 + d/2,\theta)$',fontsize=30);
	plt.legend(fontsize=25)
	plt.xlim([0,np.pi])
	plt.grid()
	plt.savefig("T_polar_Vs_theta.eps", format='eps', dpi=1800)
	plt.show()

def Norm_v_Ra(Y):

	N = len(Y)
	X = np.zeros(N); 
	NU = np.zeros(N);
	Ra = np.zeros(N);
	npr = len(R)-2;
	TT = np.zeros(len(R))
	#dr = R[0]-R[1]; 
	A_T = -(1.0+d)/d
	for ii in xrange(N):
		X[ii] = Nu(Y[ii][0:-1])
		Ra[ii] = Y[ii][-1,0]/Ra_c; #(Y[ii][-1,0] -Ra_c)/Ra_c
		
		# Perhaps use Nu -1
		TT[1:-1] = Y[ii][npr:2*npr,0];
		
		''''
		# horizontally averaged temp profile
		if ii == 99:
			plt.figure(figsize=(12,8))
			TT[1:-1] = Y[0][npr:2*npr,0];
			xx = np.linspace(R[-1],R[0],100);
			T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
			TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
			plt.plot(R,TT,'ko',markersize=3.0)
			plt.plot(xx,T,'k-',linewidth=1.5)

			TT = np.zeros(len(R))
			TT[1:-1] = Y[ii][npr:2*npr,0];
			xx = np.linspace(R[-1],R[0],100);
			T = np.polyval(np.polyfit(R,TT,len(R)),xx) + (-A_T*np.ones(len(xx))/xx) - (1.0/d)*np.ones(len(xx))
			TT = TT + (-A_T*np.ones(len(R))/R) - (1.0/d)*np.ones(len(R))
			plt.plot(R,TT,'bo',markersize=3.0) 
			plt.plot(xx,T,'b-',linewidth=1.5)

			#plt.plot(R,((R**2)/A_T)*np.matmul(D,TT),'k--')
			plt.xlabel(r'$r$',fontsize=20);
			plt.ylabel(r'$\overline{T(r,\theta)}$',fontsize=20);
			plt.xlim([1.0,4.0])
			plt.show()
		'''

		NuM1 = ((R**2)/A_T)*np.matmul(D,TT)

		Bot = NuM1[0]

		Top = NuM1[-1]

		#print Bot - Top

		NU[ii] = Bot

	params = [0,1]
	#params, params_covariance = optimize.curve_fit(test_func, Ra[30:-1], NU[30:-1]);

	print "Const, Exponent",params;

	#sys.exit()
	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$Ra/Ra_c$',fontsize=20);
	plt.ylabel(r'$Nu$',fontsize=20);
	plt.plot(Ra,NU,'k.',linewidth=1.5,label=r'DNS');
	plt.plot(Ra[50],NU[50],'yo',markerfacecolor='none',markersize=10.0);

	plt.plot(Ra[::10],params[0]*(Ra**params[1])[::10],'bo',markerfacecolor='none',markersize=3.0,label=r'$%1.3f(Ra/Ra_c)^{%1.3f}$'%(params[0],params[1]))
	
	#plt.ylim([0,0.5])
	#plt.xlim([1,np.max(Ra)])
	plt.legend(fontsize=14)
	plt.savefig("NuVsRa.eps", format='eps', dpi=1800)
	plt.show()


	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$Ra/Ra_c$',fontsize=20);
	plt.ylabel(r'$||\psi||$',fontsize=20);
	plt.plot(Ra,X,'k.',linewidth=1.5);
	#plt.plot(Ra,X,'k.',markersize=3.0);
	#plt.ylim([0,0.5])
	#plt.xlim([1,np.max(Ra)])
	plt.savefig("PsiVsRa.eps", format='eps', dpi=1800)
	plt.show()

def Lam_v_Ra(LAM,Y):

	N = len(Y);	Ra = np.zeros(N);
	LAM_Re = np.zeros((5,N)); LAM_IM = np.zeros((5,N));

	for ii in xrange(N-1):
		ii = ii + 1;
		if (LAM[ii-1][0].real*LAM[ii][0].real) < 0.0:
			print "sign change ",ii
			#sys.exit() 
		if (LAM[ii-1][1].real*LAM[ii][1].real) < 0.0:
			print "sign change ",ii
			#sys.exit() 	
		if (LAM[ii-1][2].real*LAM[ii][2].real) < 0.0:
			print "sign change ",ii
			#sys.exit() 	
		if (LAM[ii-1][3].real*LAM[ii][3].real) < 0.0:
			print "sign change ",ii
			#sys.exit() 			
		LAM_Re[:,ii] = LAM[ii][0:5].real;
		LAM_IM[:,ii] = LAM[ii][0:5].imag;
		#print "Iteration ",ii
		#print "Ra ",Y[ii][-1,0]
		#print LAM[ii],"\n"
		
		Ra[ii] = Y[ii][-1,0];# -Ra_c)/Ra_c

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$\epsilon$',fontsize=20);
	plt.ylabel(r'$\Re{\lambda}$',fontsize=20);
	line = ['r.','g*','bs','y^','ko']
	for ii in xrange(5):
		#ii = 0;
		plt.plot(Ra,LAM_Re[ii,:],line[ii],linewidth=1.5);
	plt.show()	

	plt.figure(figsize=(8, 6))
	plt.xlabel(r'$\epsilon$',fontsize=20);
	plt.ylabel(r'$\Im{\lambda}$',fontsize=20);
	for ii in xrange(5):
		plt.plot(Ra,LAM_IM[ii,:],line[ii],linewidth=1.5);
	plt.show()

# Parameters to pass
l = 1;
d = 3.0;
Ra_c = 365.0;
Ra = Ra_c*(1+0.5)
Re_1, Re_2 = 40.0,0.0;
Pr = 10.0; #10.0;
#Pr = 0.05
Nr = 40; N_Modes = 40+1;

# Define all subsequent radial dimensions from R !!!
D,R = cheb_radial(Nr,d)
print "R ",R
print R.shape
print D.shape
#sys.exit()

# Calculate All Linear Matrices Globally
#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MR2 = np.diag(1.0/(R[1:-1]**2)); #MR2 = np.matmul(MR2,D[1:-1,1:-1])
MR3 = np.diag(1.0/(R[1:-1]**3)); 
D3 = []; D2 = [];
for ii in xrange(N_Modes):
	#DD3 = np.matmul(MR2, D3_SP(D,R,ii));
	#D3.append(DD3);
	D3.append(D3_SP(D,R,ii));
	D2.append(Stokes_D2_SP(D,R,ii));

M = M_0_SP(D,R,N_Modes);
M_inv = np.linalg.inv(M);

L0 = L_0_SP(D,R,d,Pr,0.0,N_Modes); # Ra = 0.0 here!
L1 = L_1_SP(R,d,Pr,N_Modes); # To be summed and multiplied by Ra  

Force = force_SP(Pr,R,N_Modes,d,0,1.0); # Multiply by (Re_1)^2

L_Omega = L_OM_SP(D,N_Modes,R,d,Pr,0,1.0); # Multiply by Re_1
L_Cor = L_COR_SP(D,N_Modes,R,d,Pr,0,1.0); # Multiply by Re_1
L_OM_COR = L_Omega + L_Cor;
L_Ra = L1;

DIM = 3*len(R[1:-1])*N_Modes;
Y_0 = np.zeros((DIM+1,1))
X = np.zeros((DIM,1))
print "Dim ",DIM

# Load Initial Condition

'''
#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ Time-Stepping ~~~~~~~~~~~~~~~~~~~~~

dt = 2e-04;
# 2-cell oscillations 
XX = np.load("XD_HOPF_l2_Pr10.0_Re40.0.npy"); Re_1 = 40.0; Ra = 23e03; #dt = 2.0*dt;
# 1-cell oscillations
#XX = np.load("XD_HOPF_l1_Pr10.0_Re28.0.npy"); Re_1 = 28.0; Ra = 13e03; #dt = dt

#D_o,R_o = cheb_radial(25,d);
#XX = INTERP_SPEC(R,R_o,XX);

epsilon = (Ra - Ra_c)/Ra_c; #X = Y[0:-1,0];
#50000.0 = 11 hrs simulation
ITERS = int(5e04)
X,X_VID = IMPLICIT(ITERS,l,Pr,Ra_c,epsilon,d,Nr,N_Modes,dt,Re_1,Re_2,XX);

sys.exit()
'''

#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ Steady-State ~~~~~~~~~~~~~~~~~~~~~
# All Nr = 20, N_modes = 30 +1
#Y = np.load("U2_MINUS_Pr10.npy"); Pr = 10.0; Nr = 20, N_Modes = 41
#Y = np.load("U2_PLUS_Pr10.npy"); Pr = 10.0; 
#Y = np.load("1Cell2_Pr10.npy"); Pr = 10.0;  #Nr = 25, N_Modes = 41

#Y_0 = np.load("U2C_PLUS_Pr0.05.npy"); Pr = 0.05;
#Y_0 = np.load("U2C_MINUS_Pr0.05.npy"); Pr = 0.05;
#Y = np.load("1Cell2_Pr005.npy"); Pr = 0.05;  #Nr = 25, N_Modes = 41

'''
X = np.load("Test_XD_l2_Pr10.0_Re40.npy"); Re_1 = 40.0;#
Ra = 31e03;#Y_0[-1,0];
D_o,R_o = cheb_radial(25,d);
XX = INTERP_SPEC(R,R_o,X);

Y_0[0:-1] = Netwon_It(Pr,Ra,d,Nr,N_Modes,Re_1,Re_2,XX)
Y_0[-1,0] = Ra
'''

#Y_0 = np.load("Y_2C_Re40.npy"); Re_1 = 40.0; Ra = Y_0[-1,0];

Y1_0 = np.load("Y_SP1_22k_Re40.npy"); # u_r^+
Y2_0 = np.load("Y_SP2_22k_Re40.npy"); # Mixed 1-cell unstable
Y3_0 = np.load("Y_SP3_22k_Re40.npy"); # Mixed 1-cell stable
#print Y_0.shape


Temp_Prof_ESP([Y1_0,Y2_0,Y3_0])
sys.exit()
'''
args = [Pr,d,R,N_Modes,Re_1,Re_2,Y_0,D];
L,L_inv,FF,N1,N2 = Matrices_Psuedo(*args);
A = np.matmul(M_inv,L+N1+N2)
Eig_0 = EigenVals(A)[0:5]
'''

from Projection import *
X = Y_0[0:-1]; f = 1e-11; Nth = 300; 
theta = np.linspace(f,np.pi-f,Nth); xx = np.linspace(R[-1],R[0],100);
#PSI, T, OMEGA, T_0 = Outer_Prod(R,theta,N_Modes,X,d,Re_1,Re_2,Pr)
#PSI, T, OMEGA, T_0 = SPEC_TO_GRID(R,xx,theta,N_Modes,X,d,Re_1,Re_2,Pr)
#Plot_Package_CHE(xx,theta,OMEGA,PSI,T,d,Re_1,Ra,Pr)

X = Y2_0[0:-1];
PSI, T, OMEGA, T_0 = SPEC_TO_GRID_EIG(R,xx,theta,N_Modes,X,d,Re_1,Re_2,Pr)
X = Y1_0[0:-1];
PSI_imag, T_imag, OMEGA_imag, T_0 = SPEC_TO_GRID_EIG(R,xx,theta,N_Modes,X,d,Re_1,Re_2,Pr)

# Real is plotted on the left imaginary on the right
Plot_Package_CHE_HOPF(xx,theta,OMEGA,PSI,T,OMEGA_imag,PSI_imag,T_imag,d,Re_1,Ra,Pr)

Energy(X,N_Modes,R);

sys.exit()
#np.save("1Cell_Re5_Pr10.npy",Y_0)

iters = 50; # approx 1.5 hours max
eig_freq = 1;
prev_det = 0.0
ds = 250;
sign = -1.0;
delta = 0.1; #1.0/(3*Nr*N_Modes); # Useful to make this small (but not too small!!!!)W for going around turning points
print "delta ",delta;
print "Ra init ", Ra

STR1 = "".join(['YvRa_1Param_bck40_2C_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']);
STR2 = "".join(['Lamda_1Param_bck40_2C_l',str(l),'_Pr',str(Pr),'_Re',str(Re_1),'.npy']); 

start_time = time.time()
args = [Pr,d,D,R,N_Modes,Re_1,Re_2,Y_0,Eig_0];
YvsRa,Eigs = Psuedo_Branch(*args)
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))

YvsRa = np.load(STR1);
Eigs = np.load(STR2);

#sys.exit()

#print "Ra ",YvsRa[-1][-1,0];
#Energy(X,N_Modes,R);
Norm_v_Ra(YvsRa);
Lam_v_Ra(Eigs,YvsRa);
Temp_Prof(YvsRa)

from Projection import *
f = 1e-11; Nth = 300; 
theta = np.linspace(f,np.pi-f,Nth); xx = np.linspace(R[-1],R[0],100);
#PSI, T, OMEGA, T_0 = Outer_Prod(R,theta,N_Modes,X,d,Re_1,Re_2,Pr)
PSI, T, OMEGA, T_0 = SPEC_TO_GRID(R,xx,theta,N_Modes,X,d,Re_1,Re_2,Pr); #np.matmul(M,

# Plot ksi,T
Plot_Package_CHE(xx,theta,OMEGA,PSI,T,d,Re_1,Ra,Pr)
#Plot_Package_Therm(xx,theta,OMEGA,-PSI,T,d,Re_1,Ra,Pr)

#sys.exit()'''