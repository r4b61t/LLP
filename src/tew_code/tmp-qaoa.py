import numpy as np
from numpy import tensordot, kron, ones
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint, Bounds
import networkx as nx
import community

class QAOA():
    """

    Parameters

    ----------
        
    Qs : np.ndarray,
        A 2D numpy array containing Q-values.
    
    As : np.ndarray,
        A 2D numpy array containing association matrix.
        
    e : float, default=1/20
        The weight between the bank's revenue and the cash flow 
        among bank's customers in the cost function.
        
    p : int, default=2
        The number of driving cycles in QAOA.

    Attributes

    ----------
    
    N : int,
        The number of assets.
        
    M : int,
        The number of actions.
    
    h : np.ndarray,
        A 2D numpy array containing effective onsite "magnetic" fields.
        
    J : np.ndarray,
        A 2D numpy array containing "spin-spin "interactions between assets.
        
    psi : np.ndarray,
        A rank-N tensor containing a quantum state. Each leg has dimension M.
        
    costs : np.ndarray,
        Cost function during classical optimisation. 
        
    res : scipy.optimize.OptimizeResult,
        A result from scipy.optimize.minimize.
          
    """
    
    def __init__(self, Qs, As, e, p):
        
        assert Qs.ndim == 2
        assert As.ndim == 2
        assert As.shape[0] == As.shape[1] 
        assert As.shape[0] == Qs.shape[0]
        assert isinstance(e, float) and e >= 0
        assert isinstance(p, int) and p >= 1                

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

    def optimized(self, method='COBYLA',disp=False,maxiter=50):
        
        self.costs = []
        
        params = np.random.rand(2*self.p)
        
        self.res = minimize(self.cost, params, method=method, options={'disp':False,'maxiter':maxiter})      

        
    def cost(self, params):
        
        assert len(params) == 2 * self.p
        
        self.evolve(params)
        
        psi_bra = np.copy(np.conj(self.psi))
        
        c = self.c0
        
        for i in range(self.N): 
            
            psi_ket = np.copy(self.psi)
        
            c += self.inner_product(psi_bra, self._apply_h_B_onsite(psi_ket,i))                      
                
            for ii in range(i):
                
                if self.J[i,ii] != 0:
                    
                    psi_ket = np.copy(self.psi)
            
                    c += self.inner_product(psi_bra, self._apply_h_B_coupling(psi_ket,i,ii))
        
        self.costs += [np.real(c)]
        
        return np.real(c)
                
      
    def evolve(self, params):
        
        assert len(params) == 2 * self.p 
        
        self.restart(state="ones")
        
        for t in range(self.p):
            self._apply_U_B(params[2*t])
            self._apply_U_A(params[2*t+1])
 

    def restart(self, state="zeros"):
        
        if state=="zeros":
            self.psi = np.zeros(self.M**self.N,dtype='complex')
            self.psi[0] = 1
            self.psi = np.reshape(self.psi,[self.M]*self.N) 
            
        if state=="ones":
            self.psi = np.ones(self.M**self.N,dtype='complex') / np.sqrt(self.M**self.N)
            self.psi = np.reshape(self.psi,[self.M]*self.N) 

        if state=="local":
            m = Exact(self.Qs, self.As, e=0)
            m.optimized()
            self.psi = np.zeros(self.M**self.N,dtype='complex')
            self.psi[np.argmin(m.costs)] = 1
            self.psi = np.reshape(self.psi,[self.M]*self.N) 
        
    def inner_product(self,psi_1,psi_2):
        
        return np.tensordot(psi_1, psi_2, axes=(np.arange(self.N),np.arange(self.N)))
        
    
    # Apply the total U_A 
    def _apply_U_A(self, beta_t):
    
        u = self._get_u_A(beta_t)
    
        for i in range(self.N):
        
            assert np.shape(self.psi)[i] == self.M
        
            self.psi = np.tensordot(u,self.psi,axes=(1,i))  
          

    # Apply the total U_B
    def _apply_U_B(self, gamma_t):
    
        for i in range(self.N):
        
            assert np.shape(self.psi)[i] == self.M       
        
            self._apply_u_B_onsite(gamma_t,i)
                       
            for ii in range(i):
                
                if self.J[i,ii] != 0:
            
                    self._apply_u_B_coupling(gamma_t,i,ii)   
                
                
    # Get a tight-binding operator acting on one asset 
    def _get_u_A(self, beta_t):
    
        global vs, es
    
        # Eigenstates  |E_k> = a^\dagger_k |0>
        if 'vs' not in globals():
            vs = np.exp(-1j*np.array([[ 2*np.pi*k*j/self.M for k in range(self.M)] for j in range(self.M)]))
            vs = vs/ np.sqrt(self.M)
        
        # Eigenvalues e^{-iE_k} where E_k = 2*cos(k)
        if 'es' not in globals(): 
            es = np.exp(-1j*2*np.cos(np.arange(self.M)*2*np.pi/self.M))
        
        return np.conj(vs.T).dot(np.power(es, beta_t)[:,None]*vs)   
    
    
    # Apply an onsite term in U_B 
    def _apply_u_B_onsite(self, gamma_t,i):
    
        assert i < self.N
    
        u = np.exp(-1j*(-self.h[i,:]) * gamma_t)
    
        idx = '[' +'None,'*i + ':' + ',None'*(self.N-i-1) + ']'
    
        exec('self.psi *= u'+idx)


    # Apply a coupling term in U_B 
    def _apply_u_B_coupling(self, gamma_t,i,ii):
    
        assert i>ii
    
        idx = '['+':,'*ii + '0,' + ':,'*(i-ii-1) + '0' +',:'*(self.N-i-1) + ']'
    
        exec('self.psi' + idx +'*= np.exp(-1j*(-self.J[i,ii])*gamma_t)')
        
        
    # Apply an onsite term in H_B 
    def _apply_h_B_onsite(self, psi, i):
    
        assert i < self.N
    
        u = -self.h[i,:]
    
        idx = '[' +'None,'*i + ':' + ',None'*(self.N-i-1) + ']'
    
        exec('psi *= u'+idx)
        
        return psi

    # Apply the coupling term in H_B
    def _apply_h_B_coupling(self, psi,i,ii):
    
        assert i>ii
    
        # -J * n_i * n_j
        h_B_coupling = np.zeros((self.M**2,self.M**2))
        h_B_coupling[0,0] = -self.J[i,ii]
        h_B_coupling = np.reshape(h_B_coupling,[self.M]*4)
            
        return np.tensordot(psi, h_B_coupling, axes=([ii,i],[0,1]))
    

    # Convert arg into element in Psi.    
    def _to_str(self, n):
        convertString = "0123456789ABCDEF"
        if n < self.M:
            return convertString[n]
        else:
            return self._to_str(n//self.M) + convertString[n%self.M]

    def get_str_from_index(self, n):
        Str = self._to_str(n)
        return "0"*(self.N - len(Str)) + Str

    
    # Get cost from bitstring.
    def get_cost_from_str(self, state):       
        c = 0       
        for i in range(len(state)):
            c -= self.h[i, int(state[i])]
            for ii in range(i):
                if (int(state[i]) == 0 and int(state[ii]) == 0):
                    c -= self.J[i,ii]
        return c