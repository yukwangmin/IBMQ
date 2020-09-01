
from typing import Optional, Union, List, Tuple
import logging
import numpy as np
from scipy.stats import beta
import statistics as stat

from qiskit import execute

from circuit import *
from utils import *


def _clopper_pearson_confint(counts: int, shots: int, alpha: float) -> Tuple[float, float]:
    """
    Copy of _clopper_pearson_confint() function from https://github.com/Qiskit/qiskit-aqua/blob/master/qiskit/aqua/algorithms/amplitude_estimators/iqae.py
    Reference data: Apr. 2020
    
    Compute the Clopper-Pearson confidence interval for `shots` i.i.d. Bernoulli trials.
    Args:
        counts: The number of positive counts.
        shots: The number of shots.
        alpha: The confidence level for the confidence interval.
    Returns:
        The Clopper-Pearson confidence interval.
    """
    lower, upper = 0, 1

    # if counts == 0, the beta quantile returns nan
    if counts != 0:
        lower = beta.ppf(alpha / 2, counts, shots - counts + 1)

    # if counts == shots, the beta quantile returns nan
    if counts != shots:
        upper = beta.ppf(1 - alpha / 2, counts + 1, shots - counts)

    return lower, upper


def _find_next_k(k: int, upper_half_circle: bool, theta_interval: Tuple[float, float], min_ratio: int = 2) -> Tuple[int, bool]:
    """
    Copy of _find_next_k() function from https://github.com/Qiskit/qiskit-aqua/blob/master/qiskit/aqua/algorithms/amplitude_estimators/iqae.py
    Reference data: Apr. 2020
    
    """
    
    if min_ratio <= 1:
        raise AquaError('min_ratio must be larger than 1 to ensure convergence')

    # initialize variables
    theta_l, theta_u = theta_interval
    old_scaling = 4 * k + 2  # current scaling factor, called K := (4k + 2)

    # the largest feasible scaling factor K cannot be larger than K_max,
    # which is bounded by the length of the current confidence interval
    max_scaling = int(1 / (2 * (theta_u - theta_l)))
    scaling = max_scaling - (max_scaling - 2) % 4  # bring into the form 4 * k_max + 2

    # find the largest feasible scaling factor K_next, and thus k_next
    while scaling >= min_ratio * old_scaling:
        theta_min = scaling * theta_l - int(scaling * theta_l)
        theta_max = scaling * theta_u - int(scaling * theta_u)

        if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
            # the extrapolated theta interval is in the upper half-circle
            upper_half_circle = True
            return int((scaling - 2) / 4), upper_half_circle

        elif theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
            # the extrapolated theta interval is in the upper half-circle
            upper_half_circle = False
            return int((scaling - 2) / 4), upper_half_circle

        scaling -= 4

    # if we do not find a feasible k, return the old one
    return int(k), upper_half_circle



def makeIQAECircuit(n: int, k: int, b:float, barrier=False) -> QuantumCircuit:

    # set up circuit
    q = QuantumRegister(n+1, 'q')
    #a = QuantumRegister(n-2, 'a')
    c = ClassicalRegister(1, name='c0')
    qc = QuantumCircuit(q, c, name='IAE')

    # A build
    addA(qc, q, n, b, barrier)
    qc.barrier(q)

    # add Q^k
    if k != 0:
        for i in range(k):
            addQ(qc, q, n, b, barrier)

    if barrier:
        qc.barrier()
        
    qc.measure(q[0], c[0])

    return qc




def iqae(n: int, b:float, shots: int, error: float, backend, printData = True):
    
    # initialize memory variables
    powers = [0]  # list of powers k: Q^k, (called 'k' in paper)
    ratios = []  # list of multiplication factors (called 'q' in paper)
    theta_intervals = [[0, 1 / 4]]  # a priori knowledge of theta / 2 / pi
    a_intervals = [[0, 1]]  # a priori knowledge of the confidence interval of the estimate a
    num_oracle = 0
    num_one_shots = []
    min_ratio = 2.0
    alpha = 0.1
    epsilon = 0.1


    # maximum number of rounds
    max_rounds = int(np.log(min_ratio * np.pi / 8 / epsilon) / np.log(min_ratio)) + 1
    upper_half_circle = True  # initially theta is in the upper half-circle

    
    num_iterations = 0  # keep track of the number of iterations
    circuits = []
    #results = []
    counts = []
    
    # do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
    while theta_intervals[-1][1] - theta_intervals[-1][0] > error:
        num_iterations += 1


        # get the next k
        k, upper_half_circle = _find_next_k(powers[-1], upper_half_circle, theta_intervals[-1], min_ratio=min_ratio)

        if printData:
            print('======================================================')
            print('num_iterations: ', num_iterations, ', k:', k)

        # store the variables
        powers.append(k)
        ratios.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))

        # run measurements for Q^k A|0> circuit
        circuit = makeIQAECircuit(n, k, b)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        count = result.get_counts(circuit)

        circuits += [circuit]
        #results += [result]
        counts += [count]

        one_counts = count.get('1', 0)
        prob = one_counts / shots
        num_one_shots.append(one_counts)

        # track number of Q-oracle calls
        if k==0:
            num_oracle += shots
        else:
            num_oracle += shots * (2*k + 1)
            

        # if on the previous iterations we have K_{i-1} == K_i, we sum these samples up
        j = 1  # number of times we stayed fixed at the same K
        round_shots = shots
        round_one_counts = one_counts
        if num_iterations > 1:
            while powers[num_iterations - j] == powers[num_iterations] and num_iterations >= j + 1:
                j = j + 1
                round_shots += shots
                round_one_counts += num_one_shots[-j]


        a_i_min, a_i_max = _clopper_pearson_confint(round_one_counts, round_shots, alpha / max_rounds)

        # compute theta_min_i, theta_max_i
        if upper_half_circle:
            theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
            theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
        else:
            theta_min_i = 1 - np.arccos(1 - 2 * a_i_max) / 2 / np.pi
            theta_max_i = 1 - np.arccos(1 - 2 * a_i_min) / 2 / np.pi

        # compute theta_u, theta_l of this iteration
        scaling = 4 * k + 2  # current K_i factor
        theta_u = (int(scaling * theta_intervals[-1][1]) + theta_max_i) / scaling
        theta_l = (int(scaling * theta_intervals[-1][0]) + theta_min_i) / scaling
        theta_intervals.append([theta_l, theta_u])
        
        if printData:
            print('[theta_l, theta_u]: [',theta_l, ',', theta_u, ']')
            print('theta difference: ', theta_u - theta_l)

        # compute a_u_i, a_l_i
        a_u = np.sin(2 * np.pi * theta_u)**2
        a_l = np.sin(2 * np.pi * theta_l)**2
        a_intervals.append([a_l, a_u])
        
        if printData:
            print('[a_l, a_u]: [',a_l, ',', a_u, ']')
            print('a difference: ', a_u - a_l)
            print('prob:', prob)


    ##########################################################################

    if printData:
        print('======================================================')
        print('Loop Done')

    # get the latest confidence interval for the estimate of a
    a_confidence_interval = a_intervals[-1]

    # the final estimate is the mean of the confidence interval
    value = np.mean(a_confidence_interval)

    return value, num_oracle, num_iterations




def stat_iqae(n: int, b:float, shots: int, error: float, trial: int, backend, true_a: float):

    est_a = []
    rel_err_a = []
    num_oracles = []
    num_iters = []
    
    

    for i in range(trial):
        a, num_oracle, num_iter = iqae(n, b, shots, error, backend, printData=False)
        rel_a = relativeError(true_a, a)
        print('relative error or a: ', rel_a)
        print('Num oracles: ', num_oracle)

        est_a += [a]
        rel_err_a += [rel_a]
        num_oracles += [num_oracle]
        num_iters += [num_iter]
        

    print('================================================')
        
    print('Average of a: ', stat.mean(est_a))
    print('Average of relative error of a: ', stat.mean(rel_err_a))
    print('Average of oracle: ', stat.mean(num_oracles))
    print('Average of iteration: ', stat.mean(num_iters))
    
    print('Stdev of a: ', stat.stdev(est_a))
    print('Stdev of relative error of a: ', stat.stdev(rel_err_a))
    print('Stdev of oracle: ', stat.stdev(num_oracles))
    print('Stdev of iteration: ', stat.stdev(num_iters))
    
    print('Min of a: ', min(est_a))
    print('Min of relative error of a: ', min(rel_err_a))
    print('Min of oracle: ', min(num_oracles))
    print('Min of iteration: ', min(num_iters))    

    print('Max of a: ', max(est_a))
    print('Max of relative error of a: ', max(rel_err_a))
    print('Max of oracle: ', max(num_oracles))
    print('Max of iteration: ', max(num_iters))    
    
