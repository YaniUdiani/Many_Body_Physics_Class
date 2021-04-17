#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:53:40 2021

@author: YaniUdiani
"""
import itertools as tools #for infinite matter states
import numpy as np
import copy as copy
from numpy import linalg as LA


def powerset(iterable): #Function from itertools to find powerset

  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  
  s = list(iterable)
  return tools.chain.from_iterable(tools.combinations(s, r) 
                                   for r in range(len(s)+1)) 
  


def process_many_bd_state(a, Num_particles):
  
  
  all_pairs_of_sigma_minus = []; all_pairs_of_sigma_plus = []
  
  
  for i in range(Num_particles):
    
    for j in range(i+1, Num_particles):
    
      if a[i] % 2 + a[j] % 2 == 0: #pairs at sigma = -1
        
        all_pairs_of_sigma_minus.append((i,j)) #store location of 2 particles
        
      
      if a[i] % 2 + a[j] % 2 == 2: #pairs at sigma = +1
        
        all_pairs_of_sigma_plus.append((i,j)) #store location of 2 particles   
  
  
  #print(all_pairs_of_sigma_minus,all_pairs_of_sigma_plus, a)
  return all_pairs_of_sigma_minus, all_pairs_of_sigma_plus
  



def get_Ham(slater_determinants, Num_particles):
  
  
  Ham = np.zeros((len(slater_determinants), len(slater_determinants)), )
  
  
  for i1, state in enumerate(slater_determinants):
    
    
    for sp in state:
      
      Ham[i1, i1] += -(-1)**(sp%2) * epsilon/2
    
    down, up = process_many_bd_state(state, Num_particles)
    
    
    for (i, j) in down:
      
      temp_state = copy.copy(state)
      
      temp_state[i] += 1; temp_state[j] += 1 #raise pair of particles
      
      if temp_state in slater_determinants:
        
        i2 = slater_determinants.index(temp_state)
        Ham[i1, i2] -= V
        
        #print(state, temp_state)
      
  
  
    for (i, j) in up:
      
      temp_state = copy.copy(state)
      
      temp_state[i] -= 1; temp_state[j] -= 1 # lower pair of particles
      
      if temp_state in slater_determinants:
        
        i2 = slater_determinants.index(temp_state)
        Ham[i1, i2] -= V
        
        #print(state, temp_state)  
        
  
        
  return Ham





def get_slater_determinants(Omega):

  #list of all single particle states in the system
  states = list(range(2*Omega))
  
 #Store slater determinants(SD) as tuples each corresponding to a particular SD
  slater_determinants = [] 
  
  
  for SD in powerset(states):
    
    if(len(SD) == Num_particles):#grab the SDs that have Num_particles occupied
      
    #note that SD = (a,b) & a or b = where actual particle is sitting inside SD
      slater_determinants.append(list(SD))
      
      
  return slater_determinants     
  


Omega = 2
Num_particles = 2
epsilon = 1
V = 0.1
  

slater_determinants = get_slater_determinants(Omega)



Ham = get_Ham(slater_determinants, Num_particles)

spectra, vectors = LA.eig(Ham)
print("Spectra using full Slater determinant CI (Omega = N = 2): ", spectra)





######################### Problem 2 ##############################

def quasispin(K, epsilon, V):
  
  dim = 2*K+1
  proj = list(range(-K, K+1)) #get quasi-spin projections
  Ham_block = np.zeros((dim, dim), )#ready hamiltonian over projections
  
  
  for i in range(dim):
    
    Ham_block[i, i] = epsilon * proj[i]
    #print(i, proj[i], Ham_block[i, i] )
    

    if abs(proj[i] + 2) <= K:
      
      j = i + 2
      #print("upper", proj[i], proj[i + 2], i, i + 2)
      
      Ham_block[i, j] -= 0.5 * V * (np.sqrt(K*(K + 1) - proj[i]*(proj[i] + 1)) 
      * np.sqrt(K*(K + 1) - (proj[i] + 1) * (proj[i] + 2)) )
      
      
    
    if abs(proj[i] - 2) <= K:
      j = i - 2
      #print("lower", proj[i], proj[i - 2], i, i - 2)
      
      Ham_block[i, j] -= 0.5 * V * (np.sqrt(K*(K + 1) - proj[i]*(proj[i] - 1)) 
      * np.sqrt(K*(K + 1) - (proj[i] - 1) * (proj[i] - 2)) )
      
      
  return  Ham_block  

if __name__ == "__main__":
  
  for K in range(6):
    
    Quasi_Ham = quasispin(K, epsilon, V)  
    
    #print(Quasi_Ham)
  
  
    print("Spectra using quasi-spin formalism with K =  ",K,
          LA.eig(Quasi_Ham)[0])
    print("")  
  
  

