import pulp as pl
import pandas as pd
import numpy as np


def read_data(filepath):
    d = pd.read_excel(filepath, 'd', header=None)
    p = pd.read_excel(filepath, 'p', header=None)
    return d, p


number_of_centers = 4
prob_treshold = 0.6

# Read data
d_i_j = read_data("data.xlsx")[0]
p_i_j = read_data("data.xlsx")[1]
no_of_villages = d_i_j.shape[0]
x_i = np.empty(no_of_villages, dtype=pl.LpVariable)
y_i_j = np.empty((no_of_villages, no_of_villages), dtype=pl.LpVariable)

# Decision variables
for i in range(no_of_villages):
    x_i[i] = pl.LpVariable("x_"+str(i+1), cat=pl.LpBinary)
    for j in range(no_of_villages):
        y_i_j[i][j] = pl.LpVariable(
            "y_"+str(i+1)+"_"+str(j+1), cat=pl.LpBinary)

d_max = pl.LpVariable("d_max")
p_max = pl.LpVariable("p_max")

# Problem
prob = pl.LpProblem("partA", pl.LpMinimize)

# Objective function
prob += d_max

# Constraints
prob += p_max == prob_treshold

for i in range(no_of_villages):
    prob += pl.lpSum([y_i_j[i][j] for j in range(no_of_villages)]) == 1
    for j in range(no_of_villages):
        prob += y_i_j[i][j] <= x_i[j]
        prob += d_max >= d_i_j[i][j] * y_i_j[i][j]
        prob += p_i_j[i][j] * y_i_j[i][j] - p_max != 0
        prob += p_i_j[i][j] * y_i_j[i][j] <= p_max

prob += pl.lpSum([x_i[i] for i in range(no_of_villages)]) == number_of_centers

# Solve
status = prob.solve(pl.CPLEX_PY(msg=0))

if (status == 1):
    # If there is an optimal solution print the result
    center_list = []

    for v in prob.variables():
        if (v.varValue == 1 and v.name[0] == "x"):
            center_list.append(int(v.name[2:]))

    center_list.sort()
    print("Centers are at the villages with numbers", end='')
    for i in center_list:
        print(" " + str(i), end='')

    print(".\nMinimum longest distance is " +
          str(prob.objective.value()) + ".")
else:
    print("No optimal solution.")
