import numpy as np

from ibmq_utils import generate_random_dataset
from qtft_tools import *
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Q-Learning values. (Discounted Payoff action is in first column).
Qs = sparse.load_npz('data/Qs.npz').toarray()

# Association matrix
As = sparse.load_npz('data/As.npz').toarray()

# Provision constraints
Ls = sparse.load_npz('data/Ls.npz').toarray()

Qs_with_4_actions = Qs.copy()[:,:4]
Ls_with_4_actions = Ls.copy()[:,:4]

Qs = Qs_with_4_actions
Ls = Ls_with_4_actions

# np.random.seed(8)
# As, Qs = generate_random_dataset(4, 300, 0.9955138888888889)
# Ls = Qs

# Number of assets
N = len(Qs)

# Number of actions
M = len(Qs[0, :])
print(N, M)

# Number of candidate solutions from QAOA
n_candidates = 100

# Number of driving cycles in QAOA. The larger this value, the better and the slower the solutions 
p = 1

# Max community size in the divide-and-conquer algorithm. 
max_community_size = 6

# Weight (epsilon) in the objective function
e = 0.2

# Peform DC QAOA
print('-- Performing DC-QAOA --')
m = DC_QAOA(Qs, As, e, p, n_candidates, max_community_size)
m.optimized(maxiter=20)

# Perform Greedy Provision Reduction on top of DC QAOA solutions
print('-- Performing GPR on top of DC QAOA --')
g_q = GreedySearch(m.x_best, Qs, As, Ls, e=e)
g_q.optimized(300)

# Perform Greedy Provision Reduction without QAOA
print('-- Performing GPR alone for comparison --')
x = ''.join([str(np.random.randint(M)) for _ in range(N)])
g_c = GreedySearch(x, Qs, As, Ls, e=e)
g_c.optimized(300)

# Plot group sizes
fig = plt.figure(figsize=(8, 3), dpi=200)
n_groups = np.array([group.size for group in m.groups])
x, y = np.unique(n_groups, return_counts=True)
plt.plot(x, y, 'o', markersize=13)
plt.xlabel("Number of nodes", fontsize=14)
plt.ylabel("Number of Groups", fontsize=14)
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("pics/nodes.png", dpi=400)

# Plot node degree
n_degrees = []
for group in m.groups:
    n_degrees += np.sum(As[group.nodes][:, group.nodes] > 0, axis=1).tolist()

n_degrees = np.array(n_degrees)
x, y = np.unique(n_degrees, return_counts=True)

fig = plt.figure(figsize=(8, 3))
plt.plot(x, y, 'o', markersize=13)
plt.xlabel("Edge Degrees", fontsize=14)
plt.ylabel("Number of Nodes", fontsize=14)
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("pics/edges.png", dpi=400)

# Visualize a subgraph
fig = plt.figure(figsize=(8, 3))
idx = m.groups[0].nodes
g = nx.convert_matrix.from_numpy_array(As[idx][:, idx])
nx.draw(g)
plt.tight_layout()
plt.savefig("pics/example_graph.png", dpi=400)

# Plot QAOA accuracy for each group
exact_energies = np.array([group.c_exact for group in m.groups])
ind = np.argsort(exact_energies)[::-1]
fig = plt.figure(figsize=(8, 3))
for i in range(len(m.groups)):
    plt.scatter([i] * len(m.groups[ind[i]].cs), m.groups[ind[i]].cs, c='b', alpha=0.2, s=10)

plt.plot(np.arange(len(m.groups)), exact_energies[ind], 'r--', linewidth=3)
plt.ylim([-3.5, 0.2])
plt.xlabel("Group Index, sorted by energy", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.title("Candidate Distribution", fontsize=14)
_ = plt.xticks(fontsize=14)
_ = plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("pics/QAOA.png", dpi=400)

# Plot Y 
fig = plt.figure(figsize=(10, 5))
plt.plot(g_c.ys, '--', c='dimgrey', linewidth=3, label='Greedy')
plt.plot(g_q.ys, '-', c='forestgreen', linewidth=3, label='Greedy + DC QAOA')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("k", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.tight_layout()
plt.savefig("pics/Y.png", dpi=400)

# Plot L
fig = plt.figure(figsize=(10, 5))
plt.plot(g_c.ls, '--', c='dimgrey', linewidth=3, label='Greedy')
plt.plot(g_q.ls, '-', c='forestgreen', linewidth=3, label='Greedy + DC QAOA')
plt.legend(fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("k", fontsize=14)
plt.ylabel("Provision", fontsize=14)
plt.tight_layout()
plt.savefig("pics/L.png", dpi=400)

print('-- Done --')
