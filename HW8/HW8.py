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
  
  
  return LA.eig(Ham)[0],  LA.eig(Ham)[1]
  
  
  

def main():
  
  
  Omega = 10
  V = 0.1
  epsilon = 1
  chi = V * (Omega - 1)/epsilon


  subspace_H = subspace_hamiltonian(chi, Omega, epsilon)
  
  eigenValues, eigenVectors = eigendecomposition(subspace_H)
  
  print("eigenvalues of subspace Hamiltonian: ", eigenValues)
  
  print("")
  
  print("eigenvectors of subspace Hamiltonian: ", eigenVectors)
  
  
  
  plot_exact_egs = []
  plot_gcm_egs = []
  chi_s = np.linspace(0, 2, 8)
  
  
  for chi in chi_s:
    
    
    
    
    #Get GCM ground state energy
    subspace_H = subspace_hamiltonian(chi, Omega, epsilon)
  
    eigenValues, eigenVectors = eigendecomposition(subspace_H)
    
    
    GCM_e_gs = np.min(eigenValues).real
    
    
    plot_gcm_egs.append(GCM_e_gs)
    
    
    
    #Get exact ground state energy
    Quasi_Ham = Lipkin_Model1.quasispin(int(Omega/2), epsilon,
                                        epsilon * chi/(Omega-1))
    
    
    
    eigenValues, eigenVectors = eigendecomposition(Quasi_Ham)
    
    exact_e_gs = np.min(eigenValues)  
    
    plot_exact_egs.append(exact_e_gs)
    
    
    
  plt.plot(chi_s, np.transpose(plot_exact_egs), label = "EXACT")
  plt.plot(chi_s, plot_gcm_egs, label = "GCM")
  
  plt.title("GCM energies for Lipkin Model")
  plt.legend(loc ='best')   
  plt.xlabel("$\chi{}$")
  plt.ylabel("Ground State Energy")  
  plt.grid()
  plt.tight_layout()
  plt.savefig( "GCM_Energies"+ ".png" , format = "png", dpi = 500, 
            bbox_inches='tight') 
    
    
  
  
  
  
  
  
  return



if __name__ == "__main__":
  
  main()
  
  
        
      
      
 
      

      
      
  
  
  
  

  
  
  