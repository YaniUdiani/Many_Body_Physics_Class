#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 08:53:34 2021

@author: yaniudiani
"""

from scipy import integrate
import numpy as np
import scipy.special
from numpy import linalg as LA
import matplotlib.pyplot as plt # Plotting stuff
import Lipkin_Model1
import sys


def bounds_y(a,b,c,d,e,f):
    return [-np.pi, np.pi]


def bounds_x(y,a,b,c,d,e,f):
    return [-np.pi, np.pi]
  


def Hamiltonian(theta, theta_p, chi, Omega, epsilon):
  
  prefactor = -0.5 * epsilon * Omega * (np.cos(0.5*(theta - theta_p)))**Omega
  
  first_term = np.cos(0.5*(theta + theta_p)) / np.cos(0.5*(theta - theta_p))
  
  second_term = 0.5 * chi * (
  (1 + np.sin(0.5*(theta + theta_p))**2) / np.cos(0.5*(theta - theta_p))**2 -1)
  
  
  return prefactor * (first_term  + second_term)
  
  


def real_integrand(theta, theta_p, chi, Omega, epsilon, k, k_p, const):
  
  
  real_phases = (np.cos(k * theta) * np.cos(k_p * theta_p) + 
                np.sin(k * theta) * np.sin(k_p * theta_p))
  
  mat_elem = Hamiltonian(theta, theta_p, chi, Omega, epsilon)
  
  
  
  return const * real_phases * mat_elem



def imag_integrand(theta, theta_p, chi, Omega, epsilon, k, k_p, const):
  
  imag_phases = (np.sin(k * theta) * np.cos(k_p * theta_p) - 
            np.cos(k * theta) * np.sin(k_p * theta_p))
  
  mat_elem = Hamiltonian(theta, theta_p, chi, Omega, epsilon)
  
  
  return const * imag_phases * mat_elem







def administrator(chi, Omega, epsilon, k, k_p):
  
  
  n_k = (2 * np.pi/(2**Omega) ) * scipy.special.binom(Omega, k + 0.5 * Omega)
  
  n_k_p = (2 * np.pi/(2**Omega) )*scipy.special.binom(Omega, k_p + 0.5 * Omega)
  
  const = 1/(2 * np.pi * np.sqrt(n_k * n_k_p))
  
  
  real_integral_val = integrate.nquad(real_integrand, [bounds_x, bounds_y],
                                  args=(chi, Omega, epsilon, k, k_p, const))[0]
  
  imag_integral_val = integrate.nquad(imag_integrand, [bounds_x, bounds_y], 
                                  args=(chi, Omega, epsilon, k, k_p, const))[0]
  
  return (real_integral_val + imag_integral_val*1j) 

  


def subspace_hamiltonian(chi, Omega, epsilon):
  
  
  dim = Omega + 1
  
  k_values = np.linspace(-0.5 * Omega, 0.5 * Omega, num = dim)
  
  subspace_H = np.zeros((dim, dim), dtype = np.cdouble)
  
  
  for i in range(dim):
    
    for j in range(dim):
          
      subspace_H[i, j] = administrator(chi, Omega, epsilon, k_values[i], 
                                       k_values[j])
             
      
  return subspace_H


def eigendecomposition(Ham):
  
  decomp = LA.eig(Ham)
  
  return decomp[0],  decomp[1]
  
  
  
def transform_collective_wavefuctions(g, Omega, theta):

  
  dim = Omega + 1
  
  k_values = np.linspace(-0.5 * Omega, 0.5 * Omega, num = dim)

  real = 0
  
  imag = 0
  
  
  for i in range(dim):
    
    gr = g[i].real
    gi = g[i].imag
    
    
    real += gr * np.cos(k_values[i] * theta) + gi * np.sin(k_values[i] * theta)
    imag += gi * np.cos(k_values[i] * theta) - gr * np.sin(k_values[i] * theta)
    
    
  return real/np.sqrt(2*np.pi), imag/np.sqrt(2*np.pi)
  



def get_collective_wavefuctions_over_theta(g, Omega, theta_vals ):
  
  
  
  real_vals_over_theta = []
  
  imag_vals_over_theta = []
  
  
  for theta in theta_vals:
    
    compute = transform_collective_wavefuctions(g, Omega, theta)
    
    real_vals_over_theta.append(compute[0])
    imag_vals_over_theta.append(compute[1])
    
    
    
  return real_vals_over_theta, imag_vals_over_theta

    
    


def main():
  
  
  Omega = 10
  V = 0.1
  epsilon = 1
  chi = V * (Omega - 1)/epsilon


  subspace_H = subspace_hamiltonian(chi, Omega, epsilon)
  
  eigenValues, eigenVectors = eigendecomposition(subspace_H)
  
  print("eigenvalues of subspace Hamiltonian: ", eigenValues)
  
  print("")
  
  #print("eigenvectors of subspace Hamiltonian: ", eigenVectors)
  
  # for ik in range(Omega+2):
  #   print(LA.norm(eigenVectors[ik,:]))
  
  #sys.exit()
  
  plot_exact_egs = []
  plot_gcm_egs = []
  #chi_s = np.linspace(0, 2, 8)
  chi_s = np.array([0.9 , 1.00, 2.0  ])
  
  store_eigenvectors = [ [], [] , []]
  theta_vals = np.linspace(-np.pi, np.pi, 300)
  

  
  for jj in range(len(chi_s)):
    
    chi = chi_s[jj]     
    
    #Get GCM ground state energy
    subspace_H = subspace_hamiltonian(chi, Omega, epsilon)
  
    eigenValues, eigenVectors = eigendecomposition(subspace_H)
    
    print("For chi = ", chi)
    #print(""); print(eigenValues);print(np.where(eigenValues == eigenValues.min())[0][0])
    
    eigenValues = eigenValues.real
    
    
    
    #get locations of gs, first and second excited collective wavefuctions
    ind_egs = np.where(eigenValues == sorted(eigenValues)[0])[0][0]
    
    ind_first_exc = np.where(eigenValues == sorted(eigenValues)[1])[0][0]
    
    ind_second_exc = np.where(eigenValues == sorted(eigenValues)[2])[0][0]
    
    
    print("");  print(eigenValues); 
    print(""); print([ind_egs , ind_first_exc, ind_second_exc])
    
    store_eigenvectors[jj].append(eigenVectors[:, ind_egs])
    store_eigenvectors[jj].append(eigenVectors[:, ind_first_exc])
    store_eigenvectors[jj].append(eigenVectors[:, ind_second_exc])
    

  
  plt.figure()
  store_im = [ [], [] , []]
  
  for j in range(len(chi_s)):
    
    chi = chi_s[j] 
    
    for k in range(2):
      
      wavefuctions_over_theta = get_collective_wavefuctions_over_theta(
        store_eigenvectors[j][k], Omega, theta_vals )
      
      plt.plot(theta_vals, wavefuctions_over_theta[0], label = 
               "lvl = " + str(k) + ", chi = " + str(chi))
      
      #store_im[j].append(wavefuctions_over_theta[1])
      
      
  plt.title("GCM Real Collective Wavefunctions for Lipkin Model")
  plt.legend(loc ='best')   
  plt.xlabel('$\Theta$')
  plt.ylabel('$\Re [\, g(\Theta)\,]$')  
  plt.grid()
  plt.tight_layout()    
  plt.savefig( "GCM_Real_Wavefunctions_Updated"+ ".png" , format = "png", dpi = 500, 
            bbox_inches='tight') 
  
  
  
  plt.figure()
  for j in range(len(chi_s)):
    
    chi = chi_s[j] 
    
    for k in range(2):
      
      wavefuctions_over_theta = get_collective_wavefuctions_over_theta(
        store_eigenvectors[j][k], Omega, theta_vals )
      
      plt.plot(theta_vals, wavefuctions_over_theta[1], label = 
               "lvl = " + str(k) + ", chi = " + str(chi))
      
      #store_im[j].append(wavefuctions_over_theta[1])
      
      
      
      
  plt.title("GCM Imaginary Collective Wavefunctions for Lipkin Model")
  plt.legend(loc ='best')   
  plt.xlabel('$\Theta$')
  plt.ylabel('$\Im [\, g(\Theta)\,]$')  
  plt.grid()
  plt.tight_layout()    
  plt.savefig( "GCM_Imag_Wavefunctions_Updated"+ ".png" , format = "png", dpi = 500, 
            bbox_inches='tight')   
    
    
  
  
  
  
  
  
  return



if __name__ == "__main__":
  
  main()
  
  
        
      
      
 
      

      
      
  
  
  
  

  
  
  
