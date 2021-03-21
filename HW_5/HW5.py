#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:37:46 2021

@author: YaniUdiani
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import Lipkin_Model1
from numpy import linalg as LA



def dE_dtheta1(LM_parameters, tol, theta, phi):
  
  
  Omega   = LM_parameters["Omega"] 
  epsilon = LM_parameters["epsilon"]
  rrgs    = LM_parameters["rrgs"]
  chi     = LM_parameters["chi"]
  

  
  #First term
  A_1 = (-epsilon * Omega**2 * rrgs * np.sin(theta) * np.cos(theta)**(Omega-1)
       /(1 +  rrgs * np.cos(theta)**(Omega) )**2 )
  
  B_1 = np.cos(theta)**(Omega-1) 
  
  C_1 = (2* chi * (1- np.tan(theta/2)**(2))**(Omega-2) * np.tan(theta/2)**(2) *
       np.cos(theta/2)**(2*Omega)) * np.cos(2* phi)
  
  
  
  
  #Second term
  A_2 = (-epsilon * Omega /(1 +  rrgs * np.cos(theta)**(Omega) ) )
  
  B_2 = (1-Omega) * np.cos(theta)**(Omega-2) * np.sin(theta)
  
  C_2 = ( (chi/2) * np.sin(theta/2) * (1/np.cos(theta)**(3) ) * 
         np.cos(theta/2)**(2*Omega + 1) * (Omega * (np.cos(2*theta) - 1) + 4)
      * (np.cos(theta) * (1/np.cos(theta/2)**(2)) )**(Omega) ) * np.cos(2* phi) 
  
  
  val = A_1 * (B_1 + C_1) + A_2 * (B_2 + C_2)
  
  if val <= tol:
    
    print("theta_initial_guess = ", theta)
    
  
  
  return val
  


def E_r(theta, phi, LM_parameters, tol):
  
  Omega   = LM_parameters["Omega"] 
  epsilon = LM_parameters["epsilon"]
  rrgs    = LM_parameters["rrgs"]
  chi     = LM_parameters["chi"]
  

  
  #Scaling factor
  A_0 = -epsilon * Omega/(2 * (1 +  rrgs * np.cos(theta)**(Omega) ))
  
  #First term
  A_1 = np.cos(theta) + (chi/2) * np.sin(theta)**2 * np.cos(2*phi)
  
  #Second term
  A_2 = ( rrgs * (1-np.tan(theta/2)**2)**(Omega-2) * np.cos(theta/2)**(2*Omega)
         *(1 - np.tan(theta/2)**4) )
  
  #Third term
  A_3 = ( rrgs * (1-np.tan(theta/2)**2)**(Omega-2) * np.cos(theta/2)**(2*Omega)
         *( 2 * chi * np.tan(theta/2)**2 * np.cos(2*phi)) )
  

  return A_0 * (A_1 + A_2 + A_3)  
  




def dE_dtheta(LM_parameters, tol, theta, phi):
  
  
  Omega   = LM_parameters["Omega"] 
  epsilon = LM_parameters["epsilon"]
  rrgs    = LM_parameters["rrgs"]
  chi     = LM_parameters["chi"]
  

  
  #First term of left derivative
  A_1 = (-epsilon * Omega**2 * rrgs * np.sin(theta) * np.cos(theta)**(Omega-1)
       /(2 * (1 +  rrgs * np.cos(theta)**(Omega) )**2) )
  
  
  #First term of right non-derivative
  B_1 = np.cos(theta) + (chi/2) * np.sin(theta)**2 * np.cos(2*phi)
  
  #Second term of right non-derivative
  B_2 = ( rrgs * (1-np.tan(theta/2)**2)**(Omega-2) * np.cos(theta/2)**(2*Omega)
         *(1 - np.tan(theta/2)**4) )
  
  #Third term of right non-derivative
  B_3 = ( rrgs * (1-np.tan(theta/2)**2)**(Omega-2) * np.cos(theta/2)**(2*Omega)
         *( 2 * chi * np.tan(theta/2)**2 * np.cos(2*phi)) )
  
  
  val = A_1 * (B_1 + B_2 + B_3)
  
  
  #First term of left non-derivative
  C_1 = -epsilon * Omega/(2 * (1 +  rrgs * np.cos(theta)**(Omega) ))
  
  
  #First term of right derivative
  B_1 = -np.sin(theta) + chi * np.sin(theta) * np.cos(theta) * np.cos(2*phi)
  
  x = theta/2; b = 2 * chi * np.cos(2 * phi)
  
  #Second term of right derivative
  B_2 =  rrgs * ((2-Omega) * np.sin(x) * np.cos(x)**(2*Omega-3) * 
         (1 - np.tan(x)**2)**(Omega-3) * (b * np.tan(x)**2 - np.tan(x)**4 + 1))
  
  #Third term of right derivative
  B_3 = -rrgs * (Omega * np.sin(x) * np.cos(x)**(2*Omega-1) * 
         (1 - np.tan(x)**2)**(Omega-2) * (b * np.tan(x)**2 - np.tan(x)**4 + 1))
 
  #Fourth term of right derivative
  B_4 = rrgs * ( np.cos(x)**(2*Omega) * (1 - np.tan(x)**2)**(Omega-2)
         * (1/(np.cos(x))**2) *(b * np.tan(x) 
            - 2 * np.tan(x)**3 ))
  
  val += C_1 *(B_1 + B_2 + B_3 + B_4)

    
  
  # if abs(val) <= tol:
    
  #   print("theta_initial_guess = ", theta)
    
  
  
  return val



def calculate_second_derivative(val, tol, theta):
  
  
  #candidates = [for vals in val if val <= tol ]
  
  index_candidates = [ val.index(vals) for vals in val if abs(vals) <= tol ]
  
  #print([theta[k] for k in index_candidates])
  #print(index_candidates)
  for i in index_candidates:
    
    if i + 1 < len(theta):
      
      if val[i+1] - val[i] > 0: #if curvature is positive
        #print("ind", i, val[i])
        return theta[i] #return the critical value of theta
      
  



def search_root(LM_parameters, tol = 1e-1):

  
  phi = 0 #assume only one type of solution for now
  num_check = 100 #number of points to check for solution
  
  theta = np.linspace(0, np.pi/2, num_check) #grid of points of theta
  results = [dE_dtheta(LM_parameters, tol, theta_i, phi) for theta_i in theta]
  
  
  cp = calculate_second_derivative(results, tol, theta)
  print("")
  print("critial point: ", cp)  
  
  plt.figure()
  plt.plot(theta, results)
  plt.xlabel("theta")
  plt.ylabel("dE_dtheta")
  plt.axvline(x = cp)
  plt.grid()
  plt.show()
  
  
  #tau_solution = fsolve(dE_dtheta, 0, tol, theta_i, phi) 


  return cp


def Get_HF(LM_parameters):

   
  Omega   = LM_parameters["Omega"] 
  epsilon = LM_parameters["epsilon"]
  chi     = LM_parameters["chi"]
  
  
  
  if chi < 1 :
    
    return -(Omega/2) * epsilon
  
  else:
    
    return -(Omega/4) * epsilon * (chi + 1/chi)
 

   
  
################ Problem 2 ###################

#Set input parameters

Omega   = 10
epsilon = 1
rrgs    = 1
V       = 0.1
chi =  V * (Omega - 1)/epsilon
tol = 1e-1

LM_parameters = {"Omega": Omega, "epsilon": epsilon, "rrgs": rrgs, "chi": chi }


for r in [-1, 1]:
  
  LM_parameters["rrgs"] = r
  critical_point = search_root(LM_parameters, tol)
  
  if chi >= 1:  
    PAV_Egs = E_r(np.arccos(1/chi), 0, LM_parameters, tol) 
  
  else:
    PAV_Egs = E_r(0, 0, LM_parameters, tol)   
  
  print("For rrgs =  ", r, "Minimum VAP Energy :", 
         E_r(critical_point, 0, LM_parameters, tol)  )
  
  print("For rrgs =  ", r, "Minimum PAV Energy :", PAV_Egs  )  
  





################ Problems 3 & 4 ###################

#Set input parameters

Omega   = 10
epsilon = 1
rrgs    = 1
V       = 0.1
chi =  V * (Omega - 1)/epsilon
tol = 1e-1
num_pts = 30


LM_parameters = {"Omega": Omega, "epsilon": epsilon, "rrgs": rrgs, "chi": chi }

PAV = []
VAP = []
EXACT = []
HF = []
chi_s = np.linspace(0, 2, num_pts)

for chi in chi_s:
  
  LM_parameters["chi"] = chi
  critical_point = search_root(LM_parameters, tol)
  VAP_Egs = E_r(critical_point, 0, LM_parameters, tol)

  if chi >= 1:  
    PAV_Egs = E_r(np.arccos(1/chi), 0, LM_parameters, tol) 
  
  else:
    PAV_Egs = E_r(0, 0, LM_parameters, tol) 
  
  
  #print(("For chi =  ", chi, "Minimum PAV Energy :", PAV_Egs) )
  
  
  Quasi_Ham = Lipkin_Model1.quasispin(int(Omega/2),epsilon, 
                                      epsilon * chi/(Omega-1))
  Exact_Egs = LA.eig(Quasi_Ham)[0][0]
  
  HF_Egs = Get_HF(LM_parameters)

  
  PAV.append(PAV_Egs) #store PAV energies
  VAP.append(VAP_Egs)
  EXACT.append(Exact_Egs) #store exact gs energies
  HF.append(HF_Egs)
  #print(PAV_Egs, Exact_Egs, HF_Egs)
  

plt.plot(chi_s, np.transpose(EXACT), label = "EXACT")
plt.plot(chi_s, PAV, label = "PAV")
plt.plot(chi_s, VAP, label = "VAP")
plt.plot(chi_s, HF, label = "HF")
plt.title("Projected Energies for Lipkin Model")
plt.legend(loc ='best')   
plt.xlabel("$\chi{}$")
plt.ylabel("Ground State Energy")  
plt.grid()
plt.tight_layout()
#Save figure to a file for later viewing              
plt.savefig( "Projected_Energies"+ ".png" , format = "png", dpi = 500, 
            bbox_inches='tight')  

  