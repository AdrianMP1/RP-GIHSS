"""
Greedy Inclusion Algorithm.
"""
import numpy as np
from numpy.typing import NDArray
from datastructures import MaxHeap

from pymoo.indicators.hv import HV

def dominates(solution1, solution2) -> bool:
    return np.all(solution1 <= solution2) and np.any(solution1 < solution2)


def non_dominated_sorting(solutions: list) -> NDArray[np.float64]:

    solutions = np.array(solutions)

    N = solutions.shape[0]
    is_dominated = np.zeros(N, dtype=bool)

    for i in range(N):

        if is_dominated[i]:
            continue # Skip
        
        for j in range(N):
            if dominates(solutions[i], solutions[j]):
                # Solution i dominates j
                is_dominated[j] = True
            elif dominates(solutions[j], solutions[i]):
                # Solution j dominates i
                is_dominated[i] = True
                break
    
    front = solutions[~is_dominated]
    return front


def hvc_one_solution(a, others, indicator: HV) -> float:
    """
    Contribution of a in the set S.
    Others is S / a
    """

    # First, compute the worse between a and others
    modified_points = []
    for point in others:
        w = np.max([a, point], axis=0)
        modified_points.append(w)
    
    # Get unique values
    modified_points = np.unique(modified_points, axis=0)

    # Apply non dominated sorting for modified_points
    modified_points = non_dominated_sorting(modified_points)

    # Compute the contribution of a
    ## hvc = HV(a) - HV(W)
    hvc = indicator(a) - indicator(modified_points)
    return hvc

def greedy_inclusion(A_array: NDArray[np.float64], ref_point: NDArray[np.float64], mu: int):
    """
    Greedy inclusion algorithm
    
    Params.
    A: Set of all points.
    ref_point: HV parameter.
    mu: Final cardinality of set S.
    """

    S = set()
    A = set(map(tuple, A_array))

    # Make HV indicator with ref_point
    indicator = HV(ref_point)

    # Start inclusion
    while len(S) < mu:

        best_candidate = None
        best_hv = float("-inf")

        # Evaluate each candidate in A \ S
        for a in A:

            # Extend S with a
            temp_set = np.array(list(S | {a}))

            # Compute HV of S U a
            hv = indicator(temp_set)

            # Check if this candidate is the best
            if hv > best_hv:
                best_hv = hv
                best_candidate = a

        # Add the best candidate to S and remove it from A
        if best_candidate is not None:
            S.add(best_candidate)
            A.remove(best_candidate)
    
    return S


def lazy_greedy_inclusion(A_array: NDArray[np.float64], ref_point: NDArray[np.float64], mu: int):

    # Make array into a list
    A: list = A_array.tolist()

    # Initialize empty set (list)
    S: list = []

    # Initialize an empty heap
    heap = MaxHeap()
    
    # Load reference point to compute HV
    indicator = HV(ref_point)

    # Fill heap with initial conditions
    for indx, sol in enumerate(A):

        # Compute the contribution of a single solution and add it to the heap
        hvc_value = indicator(np.array(sol))
        heap.add(sol, indx, hvc_value)

    # Select the first solution
    ## Remove the solution with highest HVC
    removed_point = heap.pop()

    ## Add it to the subset and remove from A
    S.append(removed_point)
    A.remove(removed_point)

    # Start the greedy inclusion
    while len(S) < mu:

        # To numpy
        S_temp = np.array(S)

        while True:
            # Get best solutions
            last_index = heap.index[0]

            # Update its HVC
            candidate = heap.solutions[0]

            # Update the contribution of the current candidate
            new_hvc = hvc_one_solution(np.array(candidate), S_temp, indicator)
            heap.update(candidate, new_hvc)

            # Verify if the re-evaluated HVC is still the largest
            if heap.index[0] == last_index:

                # Remove candidate from heap
                removed_point = heap.pop()

                # Store in the subset
                S.append(removed_point)

                # Remove from A
                A.remove(removed_point)
                break
        
        # Save final distribution
        S_temp = np.array(S)

        return S_temp