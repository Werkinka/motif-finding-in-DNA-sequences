import argparse
import json

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Motif parameter estimator")
    parser.add_argument(
        "--input",
        default="motif_data_known_alpha.json",
        required=False,
        help="File with input data (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="estimated_params.json",
        required=False,
        help="File where the estimated parameters will be saved (default: %(default)s)",
    )
    parser.add_argument(
        "--estimate-alpha",
        choices=["yes", "no"],
        default="no",
        required=False,
        help="Should alpha be estimated? (default: %(default)s)",
    )
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha

input_file, output_file, estimate_alpha = parse_arguments()

with open(input_file, "r") as inputfile:
    data = json.load(inputfile)

X = np.asarray(data["X"])
k, w = X.shape

if estimate_alpha == "yes":
    alpha = 0.5 # TO DO: estimate alpha from X
else:
    alpha = data["alpha"]

def initialize_parameters(w):
    Theta=np.zeros((4,w))
    for j in range(w):
        x=np.random.rand(4)
        Theta[:, j]=x/np.sum(x)
    y=np.random.rand(4)
    Theta_bg=y/np.sum(y)
    return Theta, Theta_bg


def motif_prob(seq, Theta):
    p=1.0
    for j,letter in enumerate(seq):
        p*=Theta[letter-1,j]
    return p


def bg_prob(seq, Theta_bg):
    p=1.0
    for letter in seq:
        p*=Theta_bg[letter-1]
    return p


def e_step(X, Theta, Theta_bg, alpha):
    gamma=np.zeros(len(X))
    for i, seq in enumerate(X):
        p_motif=motif_prob(seq, Theta)
        p_bg=bg_prob(seq, Theta_bg)
        gamma[i]=(alpha * p_motif)/(alpha * p_motif + (1 - alpha) * p_bg)
    return gamma


def m_step(X, gamma):
    k, w = X.shape
    eps = 1e-6
    Theta = np.zeros((4, w))
    for j in range(w):
        for a in range(1, 5):
            Theta[a - 1, j] = (np.sum(gamma * (X[:, j] == a)) + eps)
        Theta[:, j] /= np.sum(Theta[:, j])
    Theta_bg = np.zeros(4)

    for a in range(1, 5):
        count = 0
        for i in range(k):
            count += ((1 - gamma[i])* np.sum(X[i] == a))
        Theta_bg[a - 1] = count + eps
    Theta_bg /= np.sum(Theta_bg)
    return Theta, Theta_bg


def log_likelihood(X, Theta, Theta_bg, alpha):
    ll = 0.0
    for seq in X:
        p_motif = motif_prob(seq, Theta)
        p_bg = bg_prob(seq, Theta_bg)
        ll += np.log(alpha * p_motif+ (1 - alpha) * p_bg)
    return ll


def run_em(X, alpha, estimate_alpha, max_iter=200,eps=1e-6):
    _, w = X.shape
    Theta, Theta_bg = initialize_parameters(w)
    prev_ll = -np.inf
    for iteration in range(max_iter):
        gamma = e_step(X,Theta,Theta_bg,alpha)
        if estimate_alpha == "yes":
            alpha = np.mean(gamma)
        Theta, Theta_bg = m_step(X,gamma)
        ll = log_likelihood(X,Theta,Theta_bg,alpha)
        if abs(ll - prev_ll) < eps:
            break
        prev_ll = ll
    return Theta, Theta_bg, alpha, ll

best_ll = -np.inf
Theta = None
Theta_bg = None
best_alpha = alpha

for _ in range(20):
    Theta_tmp, Theta_bg_tmp, alpha_tmp, ll = run_em(X,alpha,estimate_alpha)
    if ll > best_ll:
        best_ll = ll
        Theta = Theta_tmp
        Theta_bg = Theta_bg_tmp
        best_alpha = alpha_tmp

estimated_params = {
    "alpha": best_alpha,
    "Theta": Theta.tolist(),
    "Theta_bg": Theta_bg.tolist(),
}

with open(output_file, "w") as outfile:
    json.dump(estimated_params,outfile)
