#! /usr/bin/env python

# ~~~~~~~~~~~~~~~~~~~# READ ME !!!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# This code calculate a fully nonlinear solution for axisymmetric spherical convection,
# The solution is strongly influenced by initial conditions/perturbations specified within the code 

# ------ It consists of a Main and three header files:

# INV_DIFF.py ~~~~~~~~~~~~~~ returns the matrix inverse appropriate to the Crank-Nicholson Scheme employed
# Thermal_ScalarIMP.py ~~~~~ Evaluates the temperature field at the next time step
# VorticityIMP.py ~~~~~~~~~~ Evaluates the vorticity field at the next time step

# All results from the tests are stored in the chosen folder and may be plotted subsequently


#----------- Latex font ----
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.pyplot as plt
#------------------------------


import numpy as np
import sys,time,os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


from tempfile import TemporaryFile
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf) #Ensures Entire Matrix is spewed out
sys.setcheckinterval(500)

# Our Functions 
from Spec_Angle_Momentum_cy import * 
from VorticityIMP_cy import * 
from Thermal_ScalarIMP_cy import *
from INV_DIFF import *
#from Thermal_Anim import * #Plot_Package

#------ Forcing Function --------
#from forcing import *

NN = 20
azimuths = np.linspace(0,0.00001, NN)
zeniths = np.linspace(0,4.0, NN )
s = (NN,NN); ww = np.zeros(s)    

# Arrays for capturing temporal data
TORQUE, Tor_diff = np.zeros(0), np.zeros(0)
PSI_L2 = np.zeros(0); DPSI_L2 = np.zeros(0);
#A_amps, B_amps, C_amps = np.zeros(0), np.zeros(0), np.zeros(0)

#----------Set Global Parameters----------------
#epsilon = 1e-03;
Pr = 10.0;
print "Ra ",int(sys.argv[2])
Ra = int(sys.argv[2]); #20000.0; 
sigma = 3.0; # Gap width
#Ra = Ra/(sigma**3);

# ~~~~~~~~~~~ Fix the Rotation Rate ~~~~~~~~~~~~~
Re_in = int(sys.argv[1]); #30.0;
Re_out = 0.0;

# ------- Spatial Discretization Paramaters --------
R_in = 1.0; R_out = R_in*(1.0+sigma)
R_ref = R_in # Reference Length-Scale
alpha = R_out/R_ref
alpha_in = R_in/R_ref
nR = 60; nTH = 300; # Choosing the same radial resolution as Steady Sol + Eigenvector	

# -------------- Constrcut a solution grid ---------		
R = np.linspace(alpha_in,alpha,nR); dr = R[1] - R[0]
f = 1e-11; theta = np.linspace(f,np.pi-f,nTH) # No ghost nodes are used we evaluate to the wall
dth = theta[1] - theta[0];    
nr = len(R); nth = len(theta);
N = nr*nth

#print dth

dt = (1e-04)/10.0;
tsteps = 100; # Equivalent to the number of video frames being captured
Iters = 200;#02;
FAC = (Iters-2)/tsteps;
print FAC;
#print (0.3*6.28)/dt;
#sys.exit()
kk = (Pr*dt)/(dr**2); kT = (Pr*dt)/(dth**2)
print 'Stability Check R %3.4f , Theta %3.4f. Are both numbers < 0.5 ?' % (kk, kT) 

print "#-----------------------------#"
print "Prandtl Pr %6.3f [s], Rayleigh %5.3f [s] "% (Pr,Ra)
print "Re_1 %6.2f "%Re_in
print "Non-dimensional dt %1.6f"%dt
print "#-----------------------------# \n"


def vector_init_Spec(R,theta,Ainv):
	s = (N,1)
	Omega = np.zeros(s)
	ksi = np.zeros(s)
	psi = np.zeros(s)
	THERM = np.zeros(s)
	#np.save(,XX);
	X = np.load("Test_l1_Re1_30_Ra_10500.npy");
	ksi[:,0],THERM[:,0],Omega[:,0] = X[:,0],X[:,1],X[:,2]	

	#psi = stream_funcG(ksi[:,0],Ainv,R,theta);
	psi[:,0] = stream_funcG_SP(ksi[:,0],A_kp,R,theta)
	

	'''
	# ------ To load from a previous initial Cond -----------------
	#os.chdir('/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/IMPLICIT_ROT/Test2_Re_130_Ra_16000/')
	Omega = np.load("OmegaSF.npy")
	ksi = np.load("VorticitySF.npy")
	psi = np.load("Stream_functionSF.npy")
	THERM = np.load("ThermalSF.npy")
	#os.chdir('/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/IMPLICIT_ROT/')
	'''
	return Omega,ksi,psi,THERM;


#------ Main Programme Loop ------------

if __name__ == '__main__':

	# Get Start Time
	start_time = time.time()

	#---------- Declare Video Vectors -------------
	OMEGA,KSI,PSI,THERM_VID = np.zeros((N,tsteps)),np.zeros((N,tsteps)),np.zeros((N,tsteps)),np.zeros((N,tsteps)) 

	s = (N,1)
	Omega_new = np.zeros(s)
	ksi_new = np.zeros(s)
	psi_new = np.zeros(s)
	THERM_new = np.zeros(s)

	err_Omega, err_ksi, err_psi, err_sigma, err_Tor = 1.0,1.0,1.0,1.0,1.0
	i,it,tol= 0,0,1e-5;

	# Obtain Inverse A Matrices Stream function
	print("Obtain A_INV: Stream function")
	#Ainv = Build_A_Matrix(ksi_new,R,theta);
	A_kp = Build_A_Matrix_SP(ksi_new,R,theta); # returns sparse non-inverted
	
	print("Obtain A_INV: Temperature")
	# Set Pr to 1 in this exmaple
	#AINV_TH = Temp_A(THERM_new,R,theta,1.0,dt); # Sparse pentadiagonal
	A_TH = Temp_A_SP(THERM_new,R,theta,1.0,dt); # returns sparse
	
	print("Obtain A_INV: Vorticity")
	#AINV_V = KSI_A(ksi_new,R,theta,Pr,dt); # PENT Very Sparse
	A_V = KSI_A_SP(ksi_new,R,theta,Pr,dt); # returns sparse

	#plt.spy(A, precision=1e-12,markersize=5)
	#plt.spy(B, precision=1e-12,markersize=5)
	#plt.show()
	#sys.exit()
	# Sparsify these matrices
	A_kp= csr_matrix(A_kp);
	A_TH = csr_matrix(A_TH);
	A_V = csr_matrix(A_V);

	# Call to initialize Vectors
	Omega,ksi,psi,THERM= vector_init_Spec(R,theta,A_kp);

	# Video Arrays
	OMEGA[:,0] = Omega[:,0]
	KSI[:,0] = ksi[:,0]
	PSI[:,0] = psi[:,0]
	THERM_VID[:,0] = THERM[:,0]

	#person = input('Are these Parameters OK: ')
	#print os.getcwd()

	max_err = np.zeros(4);
	max_err[0], max_err[1], max_err[2], max_err[3] = err_Omega, err_ksi, err_psi, err_sigma;

	try:
		#while (max(max_err) > tol):
		for i in xrange(Iters):			
			# 1) Obtain Specific_Angular_Momentum
			args_Om = [dt,R,theta,Re_in,Re_out,Omega[:,0],psi[:,0],Pr];
			Omega_new = Therm_ang_moment_Pr(*args_Om);
			Omega_new[:,0] = spsolve(A_V,Omega_new);
			#Omega_new = np.matmul(AINV_V,Omega_new);
				
			# 2) Obtain Temperature Distribution
			argsT = [dt,R,theta,Pr,THERM[:,0],psi[:,0]];
			THERM_new = Thermal_Conv_Pr_IMP(*argsT);
			THERM_new[:,0] = spsolve(A_TH,THERM_new);
			#THERM_new = np.matmul(AINV_TH,THERM_new);

			# 3) Obtain Vorticity
			args2 = [dt,R,theta,Pr,Ra,psi[:,0],Omega[:,0],ksi[:,0],THERM[:,0]];
			ksi_new = Vorticity_Thermal_Pr_IMP(*args2); 
			ksi_new[:,0] = spsolve(A_V,ksi_new);
			#ksi_new = np.matmul(AINV_V,ksi_new);

			# 4) Resolve Stream Function
			#psi_new = stream_funcG(ksi[:,0],Ainv,R,theta)
			psi_new[:,0] = stream_funcG_SP(ksi[:,0],A_kp,R,theta)
			
			#Poach the array max/min
			PSI_L2 = np.append(PSI_L2,np.linalg.norm(psi_new,2))

			if (it<(tsteps-2)) and (i%FAC == 0): 
				#------- ARRAY CAPTURE FOR VIDEO -------
				OMEGA[:,it+1] = Omega_new[:,0]
				KSI[:,it+1] = ksi_new[:,0]; 
				PSI[:,it+1] = psi_new[:,0];
				THERM_VID[:,it+1] = THERM_new[:,0];
				it+=1

			if i%10 == 0:

				# Calculate error using a relative L-2 ||norm||
				err_Omega = np.linalg.norm((Omega_new[:,0] - Omega[:,0]),2)/np.linalg.norm(Omega_new[:,0],2) # No Motion
				err_ksi = np.linalg.norm((ksi_new[:,0] - ksi[:,0]),2)/np.linalg.norm(ksi_new[:,0],2)
				err_psi = np.linalg.norm((psi_new[:,0] - psi[:,0]),2)/np.linalg.norm(psi_new[:,0],2)
				err_sigma = np.linalg.norm((THERM_new[:,0] - THERM[:,0]),2)/np.linalg.norm(THERM_new[:,0],2)			
				max_err[0], max_err[1], max_err[2], max_err[3] = err_Omega, err_ksi, err_psi, err_sigma;

				print 'Omega Error: ', err_Omega 
				print 'Ksi Error: ', err_ksi
				#print 'Psi Error: ', err_psi
				print 'SIGMA Error: ', err_sigma
				print 'Iteration: ', i
			
			ksi = ksi_new;
			psi = psi_new;
			THERM = THERM_new;
			Omega = Omega_new;

			i+=1
			
	except KeyboardInterrupt:
		pass

	# Print Out time taken
	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))		

	# --------------------------- SAVING AND COMPILING RESULTS -------------------
	#1) Change directory into Results file
	#os.chdir('/home/mannixp/Dropbox/Imperial/PhD_Thermal/CODES/IMPLICIT_ROT/')
	#os.chdir(??)
	# 2) Make new local results folder
	#st2 = "%4.1f"%a[iRe]
	stl = "%1d"%Re_in
	stm = "%1d"%Ra
	#stP = "%2.2f"%Pr
	#PP = "%4.1f"%nTH
	
	Stringy = ['Test3_Re_1',stl,'_Ra_',stm]
	os.mkdir("".join(Stringy))

	# 3) Change into Results folder
	os.chdir("".join(Stringy))	

	# Save multiple arrays for video
	np.save('OmegaVid.npy',OMEGA)
	np.save('VorticityVid.npy',KSI)
	np.save('Stream_functionVid.npy',PSI)
	np.save('Thermal_Vid.npy',THERM_VID)

	# Save Final Solution fram
	np.save("OmegaSF.npy",Omega)
	np.save("VorticitySF.npy",ksi)
	np.save("Stream_functionSF.npy",psi)
	np.save("ThermalSF.npy",THERM)

	#np.save("Torque.npy",TORQUE)
	#np.save("TorqueDiff.npy",Tor_diff)

	# Save PSI arrays
	np.save("PSI_L2.npy",PSI_L2)

	#np.save("A_amps.npy",A_amps)
	#np.save("B_amps.npy",B_amps)
	#np.save("C_amps.npy",C_amps)

	#Create an associated Parameters file
	Parameters = np.zeros(8)
	file = open("Parameters.txt",'w')
	Parameters[0] = sigma
	file.write('%5.3f'% sigma)
	file.write(" # Sigma Gap Width \n")

	Parameters[1] = Re_in #a[iRe]
	file.write('%5.4f'% Re_in)
	file.write(" # Inner Reynolds Number \n" )
	Parameters[2] = Re_out
	file.write('%5.4f'% Re_out)
	file.write(" # Outer Reynolds Number \n" )

	Parameters[3] = Ra
	file.write('%3.3f'%Ra)
	file.write(" # Rayleigh number \n")

	Parameters[4] = Pr
	file.write('%3.3f'%Pr)
	file.write(" # Prandtl number \n")

	Parameters[5] = dt
	file.write('%1.6f' % dt)
	file.write(" # Time step \n")

	Parameters[6] = nR
	file.write('%d' % nR)
	file.write(" # nr radial steps \n")
	Parameters[7] = nTH
	file.write('%d' % nTH)
	file.write(" # nth polar steps \n")

	file.close()

	np.save("Parameters.npy",Parameters)

	'''plt.plot(np.arange(0,len(PSI_L2),1)*dt,PSI_L2[:],'k.')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$||\psi||$')
	plt.grid()
	STR = "".join(['Norm_v_time.eps'])
	plt.savefig(STR, format='eps', dpi=1200)
	plt.show()
	'''