import numpy as np


def generate_random_dataset(M, N, strength=0.2):
    As = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            x = np.random.random()
            while x > strength:
                x = np.random.random()
            As[i][j] = x
            As[j][i] = x

    Qs = np.zeros((N,M))
    expected_returns = np.linspace(1/M, 1, M)[:: -1]*(1/M)
    Vs = As.sum(axis=1)
    for i in range(N):
        np.random.shuffle(expected_returns)
        Qs[i, :] = Vs[i] * expected_returns
    return As, Qs


def convert_bitstring(bitstring, M, N):
    res = ''
    for i in M*np.arange(N):
        for j in range(M):
            if int(bitstring[i+j]) == 1:
                res = str(M-1-j) + res
                break
    return res


def print_result(result, M, N):
    sorted_states = sorted(result.items(), key=lambda x: -x[1])
    print(len(sorted_states))

    print("state bitstring       probability")
    print("-----------------------------------------------------------------------------")
    for _ in range(len(sorted_states)):
        if _ > 1000:
            break
        bit_rep = bit_repr(sorted_states[_][0])
        state = convert_bitstring(bit_rep, M, N)
        print(state, bit_rep, sorted_states[_][1])


def to_prob_dict(result, M, N):
    states = result.items()
    return {bit_repr(state, N, M): prob for state, prob in states}


def if_valid_state(bitstring, M):
    valid = True
    for i in range(0, len(bitstring), M):
        valid = True
        summation = 0
        for s in bitstring[i:i+M]:
            summation += int(s)
        if summation != 1:
            valid = False
            break
    return valid


def bit_repr(state_index, N, M):
    bit_rep = bin(state_index)[2:]
    while len(bit_rep) < N*M:
        bit_rep = '0' + bit_rep
    return bit_rep


def l1_norm(ptew_prob_dict, prob_dict):
    res = 0
    for state in ptew_prob_dict.keys():
        tew_prob = ptew_prob_dict[state]
        qiskit_prob = prob_dict.get(state) if prob_dict.get(state) else 0
        res += np.abs(tew_prob - qiskit_prob)
    return res


def fidelity(ptew_prob_dict, prob_dict):
    res = 0
    for state in ptew_prob_dict.keys():
        tew_prob = ptew_prob_dict[state]
        qiskit_prob = prob_dict.get(state) if prob_dict.get(state) else 0
        res += np.sqrt(tew_prob * qiskit_prob)
    return res**2
