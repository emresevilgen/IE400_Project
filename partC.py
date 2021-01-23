import pulp as pl
import pandas as pd
import numpy as np


def read_data(filepath):
    d = pd.read_excel(filepath, 'd', header=None)
    p = pd.read_excel(filepath, 'p', header=None)
    return d, p


prob_treshold = 0.6
start_village = 1
velocity = 40

# Read data
d_i_j = read_data("data.xlsx")[0]
p_i_j = read_data("data.xlsx")[1]
no_of_villages = d_i_j.shape[0]
u_i = np.empty(no_of_villages, dtype=pl.LpVariable)
x_i_j = np.empty((no_of_villages, no_of_villages), dtype=pl.LpVariable)

# Decision variables
for i in range(no_of_villages):
    if (i != start_village - 1):
        u_i[i] = pl.LpVariable("u_"+str(i+1), lowBound=2,
                               upBound=no_of_villages, cat=pl.LpInteger)
    else:
        u_i[i] = pl.LpVariable("u_"+str(i+1), lowBound=1,
                               upBound=1, cat=pl.LpInteger)

    for j in range(no_of_villages):
        x_i_j[i][j] = pl.LpVariable(
            "x_"+str(i+1)+"_"+str(j+1), cat=pl.LpBinary)

p_max = pl.LpVariable("p_max")

# Problem
prob = pl.LpProblem("partA", pl.LpMinimize)

# Objective function
prob += pl.lpSum([x_i_j[i][j] * d_i_j[i][j] / 40
                  for i in range(no_of_villages) for j in range(no_of_villages)])

# Constraints
prob += u_i[start_village - 1] == 1
prob += p_max == prob_treshold

for i in range(no_of_villages):
    prob += pl.lpSum([x_i_j[i][j] for j in range(no_of_villages)]) == 1
    prob += pl.lpSum([x_i_j[j][i] for j in range(no_of_villages)]) == 1
    prob += x_i_j[i][i] == 0

    for j in range(no_of_villages):
        if (i != start_village - 1 and j != start_village - 1 and i != j):
            prob += u_i[i] - u_i[j] + no_of_villages * \
                x_i_j[i][j] <= no_of_villages - 1
        prob += p_i_j[i][j] * x_i_j[i][j] <= p_max

# Solve
status = prob.solve(pl.CPLEX_PY(msg=0))

if (status == 1):
    # If there is an optimal solution print the result
    path_list = []
    for v in prob.variables():
        if (v.varValue == 1 and v.name[0] == "x"):
            temp = (v.name[2:]).split('_')
            path_list.append((int(temp[0]), int(temp[1])))

    print("Path is", start_village, end='')
    curr = start_village
    for src, dest in path_list:
        if src == curr:
            curr = dest
            break
    
    while curr != start_village:
        print (" ->", curr, end='')
        for src, dest in path_list:
            if src == curr:
                curr = dest
                break
    
    print (" ->", curr, end='')

    print(".\nMinimum time is " + str(prob.objective.value()) + ".")
else:
    print("No optimal solution.")
