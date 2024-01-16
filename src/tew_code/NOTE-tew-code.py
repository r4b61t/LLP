class QAOA():
    def __init__(self, Qs, As, e, p):
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
    #def __init__(self, Qs, As, e):
    #def optimized(self):
    #def _index2bitstring(self,idx):
    #def _set_exact_energies(self):      
    #def _onsite_op(self,P,i):
    #def _twosite_op(self,P,Q,i,j):

# Groups — attributes are candidate solutions with probs & bitstrings
class Group:
    def __init__(self, name, nodes):
    #def show(self):

class DC_QAOA():
    def __init__(self, Qs, As, e = 1/20, p = 2, n_candidates = 10, max_community_size = 7):
    def optimized(self, maxiter=20, method='BFGS'):
    def set_communities(self):
    def _combine_groups(self, L, R, ps, m):
    def _combine_bitstrings(self, x_L, x_R, ind_L, ind_R, LR):
    def _get_louvian_communities_with_shared_nodes(self, group):
    
class GreedySearch():
    def __init__(self, x0, Qs, As, Ls, e = 1/20):  
    def optimized(self, maxiter = 300):
    #def update_string(self, bitstring, asset_i, action_j):
    #def get_objective_function(self, state):
    #def get_provision(self, state):

================================================================================

DC_QAOA.__init__
DC_QAOA.optimized
    DC_QAOA.set_communities
        for 
            Group.__init__
            DC_QAOA._get_louvian_communities_with_shared_nodes
    for
        QAOA.__init__
        QAOA.optimized
            ...
        for 
            QAOA.get_str_from_index
                QAOA._to_str
        for
            QAOA.get_cost_from_str
        DC_QAOA._combine_groups
            Group.__init__
            for & while
                DC_QAOA._combine_bitstrings
            QAOA.__init__
            for
                QAOA.get_cost_from_str

================================================================================

QAOA.optimized
    ...

    VVV

QAOA.optimized
    QAOA.cost
        QAOA.evolve
            QAOA.restart
            for
                QAOA._apply_U_B
                    QAOA._apply_u_B_onsite
                    for
                        QAOA._apply_u_B_coupling
                QAOA._apply_U_A
                    QAOA._get_u_A
        for
            QAOA.inner_product
            QAOA._apply_h_B_onsite
            for 
                QAOA.inner_product
                QAOA._apply_h_B_coupling

================================================================================                

Exact.__init__
Exact.optimized
    Exact._set_exact_energies
        for
            _onsite_op
            for
                _twosite_op
    Exact._index2bitstring


================================================================================                

GreedySearch.__init__
GreedySearch.optimized
    for
        update_string
        get_objective_function
        get_provision