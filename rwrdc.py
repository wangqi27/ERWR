# python code
def RWRDC(adj_matrix, c=0.05):
    n = adj_matrix.shape[0]
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    inv_degree_matrix = np.linalg.pinv(degree_matrix)
    w = inv_degree_matrix @ adj_matrix
    identity_matrix = np.eye(n)
    
    p = np.zeros((n, n))
    for k in range(n):
        r0 = np.zeros(n)
        r0[k] = 1
        p[:, k] = (1 - c) * np.linalg.pinv(identity_matrix - c * w.T) @ r0
    
    s = p + p.T
    return s