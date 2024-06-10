import numpy as np


def updateClusters(LLE_node_vals, switch_penalty=1):

    (T, num_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    # compute future costs
    for i in range(T-2, -1, -1):
        j = i+1
        indicator = np.zeros(num_clusters)
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        for cluster in range(num_clusters):
            total_vals = future_costs + lle_vals + switch_penalty
            total_vals[cluster] -= switch_penalty
            future_cost_vals[i, cluster] = np.min(total_vals)

    # compute the best path
    path = np.zeros(T)

    # the first location
    aaa = future_cost_vals[0, :] + LLE_node_vals[0, :]
    curr_location = np.argmin(aaa)
    path[0] = curr_location

    # compute the path
    for i in range(T-1):
        j = i+1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        total_vals = future_costs + lle_vals + switch_penalty
        total_vals[int(path[i])] -= switch_penalty

        path[i+1] = np.argmin(total_vals)

    # return the computed path
    return path