import numpy as np
import torch
import pandas as pd
import cvxpy as cp
from scipy import optimize 
from scipy.special import expit
from collections import defaultdict

data_path = 'data/'

################## Spectrum Kernel ############################

class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0  # Store frequency

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.leafs = []

    def insert(self, kstr):
        """Insert a k-str into the trie."""
        node = self.root
        for char in kstr:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.count += 1  # Increase count at leaf node
    
    def build(self, sample, k):
        """Build a trie from a sample."""
        for i in range(len(sample) - k + 1):
            kstr = sample[i:i+k]
            self.insert(kstr)
            self.leafs.append(kstr)

    def search(self, kstr):
        """Search for a k-str and return its frequency."""
        node = self.root
        for char in kstr:
            if char not in node.children:
                return 0  # k-str not found
            node = node.children[char]
        return node.count  # Return k-str frequency


def build_kernel_matrix_spectrum(samples, k):
    """Build a kernel matrix from samples."""
    n = len(samples['seq'])
    K = np.zeros((n, n))
    k = 3
    for i in range(n):
        trie1 = Trie()
        sample1 = samples['seq'][i]
        trie1.build(sample1, k)
        
        for j in range(i, n):
            trie2 = Trie()
            sample2 = samples['seq'][j]
            trie2.build(sample2, k)

            common_leafs = list(set(trie1.leafs) & set(trie2.leafs))
            for kstr in common_leafs:
                K[i, j] += trie1.search(kstr) * trie2.search(kstr)

            K[j, i] = K[i, j]
    return K

def test_spectrum_kernel(alpha, train_set, test_set, k):
    """Test the spectrum kernel on a test set."""
    n = len(train_set['seq'])
    m = len(test_set['seq'])
    y_pred = np.zeros(m)
    for i in range(m):
        trie1 = Trie()
        sample1 = test_set['seq'][i]
        trie1.build(sample1, k)
        
        for j in range(n):
            trie2 = Trie()
            sample2 = train_set['seq'][j]
            trie2.build(sample2, k)

            common_leafs = list(set(trie1.leafs) & set(trie2.leafs))
            for kstr in common_leafs:
                y_pred[i] += alpha[j] * trie2.search(kstr) * trie1.search(kstr)
    return y_pred
    

####################### Mismatch Kernel ############################

class TrieNode:
    """A single node in the mismatch trie."""
    def __init__(self):
        self.children = {}
        self.count = 0  # Stores k-mer frequency

class MismatchTree:
    """Trie-based structure for (k, m)-mismatch kernel computation."""
    def __init__(self, k, m, alphabet="ACGT"):
        self.root = TrieNode()
        self.k = k
        self.m = m
        self.alphabet = alphabet
    
    def insert(self, kmer):
        """Insert k-mer into the trie."""
        node = self.root
        for char in kmer:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.count += 1  # Increase k-mer frequency
    
    def search_with_mismatches(self, kmer, max_mismatches):
        """Finds all k-mers in the tree that match with at most `max_mismatches`."""
        results = []
        
        def dfs(node, index, mismatches, path):
            """Recursive depth-first search to allow mismatches."""
            if mismatches > max_mismatches:
                return
            if index == self.k:
                results.append(("".join(path), node.count))
                return
            
            char = kmer[index]
            # Try exact match
            if char in node.children:
                dfs(node.children[char], index + 1, mismatches, path + [char])
            
            # Try mismatches
            for alt_char in self.alphabet:
                if alt_char != char and alt_char in node.children:
                    dfs(node.children[alt_char], index + 1, mismatches + 1, path + [alt_char])
        
        dfs(self.root, 0, 0, [])
        return results
    
    def build_from_sequence(self, sequence):
        """Extract all k-mers from a sequence and insert them into the trie."""
        n = len(sequence)
        for i in range(n - self.k + 1):
            kmer = sequence[i:i + self.k]
            self.insert(kmer)
    
    def mismatch_kernel(self, sequence):
        """Compute mismatch kernel value for a given sequence."""
        profile = defaultdict(int)
        n = len(sequence)

        for i in range(n - self.k + 1):
            kmer = sequence[i:i + self.k]
            matches = self.search_with_mismatches(kmer, self.m)
            # print(f'For kmers {kmer} found {matches}')

            for matched_kmer, count in matches:
                profile[matched_kmer] += count
        
        return profile

def compute_scal_prod(tree, seq2):
    """Computes the (k, m)-mismatch kernel similarity between two sequences."""
    
    # Get profile for seq2 using seq1’s mismatch tree
    profile_seq2 = tree.mismatch_kernel(seq2)
    
    # Compute dot product
    dot = 0
    for kmer, count in profile_seq2.items():
        dot += count * profile_seq2[kmer]
    
    return dot

def build_kernel_matrix_mism(samples, k, m):
    """Build a kernel matrix from samples."""
    n = len(samples['seq'])
    print(n)
    # K = np.zeros((n, n))
    K = np.load('K.npy')

    for i in range(n):
        
        tree = MismatchTree(k, m)

        seq1 = samples['seq'][i]
        tree.build_from_sequence(seq1)

        for j in range(i, n):

            seq2 = samples['seq'][j]
            kernel_value = compute_scal_prod(tree, seq2)

            K[i, j] = kernel_value
            K[j, i] = kernel_value
            
        print(f'Row i = {i} and column j = {j}', end = '\r')
        if i % 10 == 0:
            np.save(f'K.npy', K)
        
    return K
    
######################################################################

def build_kernel_matrix(samples, k, m, kernel_type):
    """Build a kernel matrix from samples."""
    if kernel_type == 'spectrum':
        return build_kernel_matrix_spectrum(samples, k)
    elif kernel_type == 'mismatch':
        return build_kernel_matrix_mism(samples, k, m)
    else:
        raise ValueError('Unknown kernel type')
    
################################################################

def KRR(K, y, lambda_reg):
    n = len(y)
    alpha = np.linalg.solve(K + lambda_reg * n * np.eye(n), y)
    return alpha

##########################################################

def sigmoid(x):
    return expit(x)

def WKRR(K, W, y, lambda_reg):
    """
    Solver for weighted kernel ridge regression.

    Parameters
    ----------
    K : np.array
        The kernel matrix.
    W : np.array
        The weight matrix, diagonal positive matrix.
    y : np.array
        The target vector.
    lambda_reg : float
        The regularization parameter.

    Returns
    -------
    np.array
        The solution of the optimization problem.   
    """

    sqrt_W = np.sqrt(W)
    n = len(y)

    beta = np.linalg.solve(sqrt_W @ K @ sqrt_W + lambda_reg * n * np.eye(len(y)), sqrt_W @ y)
    alpha = sqrt_W @ beta

    return alpha

def IRLS(K, y, lambda_reg, tol = 1e-4, max_iter=10):
    """
    Solver for Iteratively Reweighted Least Squares.

    Parameters
    ----------
    K : np.array
        The kernel matrix.
    y : np.array
    """
    
    n = len(y)
    alpha0 = np.random.randn(n)*1e-4

    for i in range(max_iter):
        
        m = K @ alpha0
        P = np.diag( - sigmoid(- y*m) )
        W = np.diag( sigmoid(m) * sigmoid(-m) )
        z = m + y / sigmoid(y * m)

        alpha1 = WKRR(K, W, z, lambda_reg)

        dist = np.linalg.norm(alpha1 - alpha0, ord = np.inf)
        if dist < tol:
            break

        alpha0 = alpha1

    if i == max_iter - 1:
        print(f'IRLS reached the maximum number of iterations. The last relative error is {dist}.')
    else:
        print(f'IRLS converged after {i+1} iterations.')
    
    return alpha1

##################################################################à

def predict(K, alpha, y, method_type):

    if method_type == 'KRR':
        
        return KRR(K, y, alpha)
    
    elif method_type == 'IRLS':

        return IRLS(K, y, alpha)
    
    else:
        raise ValueError('The allowed method types are KRR and IRLS.')
    
####################################################################

def build_data_matrix(data):
    
    X1 = np.zeros((data.shape[0], 100))
    data = data.values

    for i in range(data.shape[0]):
        values = data[i][0].split(" ")

        for j in range(len(values)):
            X1[i][j] = float(values[j])

    return X1

#################################################################

# Dataset 0
Xtr0_mat100 = pd.read_csv(data_path + 'Xtr0_mat100.csv', header=None)
Xtrain0 = build_data_matrix(Xtr0_mat100)
labels_set0 = pd.read_csv(data_path + 'Ytr0.csv')
y0 = labels_set0['Bound'].values

test_set0 = pd.read_csv(data_path + 'Xte0.csv')
Xte0_mat100 = pd.read_csv(data_path + 'Xte0_mat100.csv', header=None)
Xtest0 = build_data_matrix(Xte0_mat100)

# Dataset 1
Xtr1_mat100 = pd.read_csv(data_path + 'Xtr1_mat100.csv', header=None)
Xtrain1 = build_data_matrix(Xtr1_mat100)
labels_set1 = pd.read_csv(data_path + 'Ytr1.csv')
y1 = labels_set1['Bound'].values

test_set1 = pd.read_csv(data_path + 'Xte1.csv')
Xte1_mat100 = pd.read_csv(data_path + 'Xte1_mat100.csv', header=None)
Xtest1 = build_data_matrix(Xte1_mat100)

# Dataset 2
Xtr2_mat100 = pd.read_csv(data_path + 'Xtr2_mat100.csv', header=None)
Xtrain2 = build_data_matrix(Xtr2_mat100)
labels_set2 = pd.read_csv(data_path + 'Ytr2.csv')
y2 = labels_set2['Bound'].values

test_set2 = pd.read_csv(data_path + 'Xte2.csv')
Xte2_mat100 = pd.read_csv(data_path + 'Xte2_mat100.csv', header=None)
Xtest2 = build_data_matrix(Xte2_mat100)


########################## Implementetion of best method    ############################

# KRR

# Dataset 0
K0 = Xtrain0 @ Xtrain0.T
y0 = 2*y0 - 1
alpha0 = KRR(K0, y0, 1e-4)

y_pred0 = Xtest0 @ Xtrain0.T @ alpha0
y_pred0 = (np.sign(y_pred0) + 1) / 2


# Dataset 1
K1 = Xtrain1 @ Xtrain1.T
y1 = 2*y1 - 1
alpha1 = KRR(K1, y1, 1e-4)

y_pred1 = Xtest1 @ Xtrain1.T @ alpha1
y_pred1 = (np.sign(y_pred1) + 1) / 2


# Dataset 2
K2 = Xtrain2 @ Xtrain2.T
y2 = 2*y2 - 1
alpha2 = KRR(K2, y2, 1e-4)

y_pred2 = Xtest2 @ Xtrain2.T @ alpha2
y_pred2 = (np.sign(y_pred2) + 1) / 2


# Save the predictions in a csv file

with open('results.csv', 'w') as f:
    f.write('Id,Bound\n')
    for i, y in enumerate(y_pred0):
        f.write(f'{test_set0["Id"][i]},{int(y)}\n')
    for i, y in enumerate(y_pred1):
        f.write(f'{test_set1["Id"][i]},{int(y)}\n')
    for i, y in enumerate(y_pred2):
        f.write(f'{test_set2["Id"][i]},{int(y)}\n')
