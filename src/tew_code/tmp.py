import numpy as np
from numpy import tensordot, kron, ones
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint, Bounds
import networkx as nx
import community
    
class QAOA():
    def __init__(self, Qs, As, e, p): 
        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.Qs = np.copy(Qs)
        self.As = np.copy(As)
        self.h = (1-e)*np.copy(Qs)
        self.h[:,0] += -e*np.sum(As,axis=1)
        self.J = e*np.copy(As)
        self.e = e
        self.p = p
        self.c0 = -self.e * np.sum(As)/2
        self.psi = None
        self.costs = None
        self.res = None     
    def cost(self, params):   
    def optimized(self, method='COBYLA',disp=False,maxiter=50):                      
    def evolve(self, params):
    def restart(self, state="zeros"):        
    def inner_product(self,psi_1,psi_2):
    def _apply_U_A(self, beta_t):
    def _apply_U_B(self, gamma_t):
    def _get_u_A(self, beta_t):
    def _apply_u_B_onsite(self, gamma_t,i):
    def _apply_u_B_coupling(self, gamma_t,i,ii):
    def _apply_h_B_onsite(self, psi, i):
    def _apply_h_B_coupling(self, psi,i,ii): 
    def _to_str(self, n):
    def get_str_from_index(self, n):
    def get_cost_from_str(self, state):       

class Exact():    
    def __init__(self, Qs, As, e): 
    def optimized(self): 
    def _index2bitstring(self,idx):
    def _set_exact_energies(self):
    def _onsite_op(self,P,i):
    def _twosite_op(self,P,Q,i,j):  

class Group:  
    def __init__(self, name, nodes):
    def show(self):
                

class DC_QAOA():
    def __init__(self, Qs, As, e = 1/20, p = 2, n_candidates = 10, max_community_size = 7):       
        self.Qs = Qs
        self.As = As
        self.e = e
        self.p = p
        self.n_candidates = n_candidates
        self.max_community_size = 7
        self.N = Qs.shape[0]
        self.M = Qs.shape[1]
        self.groups = []
        self.res = None
    def optimized(self, maxiter=20, method='BFGS'):
    def set_communities(self):
    def _combine_groups(self, L, R, ps, m):    
    def _combine_bitstrings(self, x_L, x_R, ind_L, ind_R, LR):
    def _get_louvian_communities_with_shared_nodes(self, group):
    
class GreedySearch():
    def __init__(self, x0, Qs, As, Ls, e = 1/20):
    def optimized(self, maxiter = 300):
    def update_string(self, bitstring, asset_i, action_j):
    def get_objective_function(self, state):    
    def get_provision(self, state):