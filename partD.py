from sys import path
import pulp as pl
import pandas as pd
import numpy as np


def read_data(filepath):
    d = pd.read_excel(filepath, 'd', header=None)
    p = pd.read_excel(filepath, 'p', header=None)
    return d, p


start_village = 1
velocity = 40
time_threshold = 10
max_distance = velocity * time_threshold

# Read data
d_i_j = read_data("data.xlsx")[0]
no_of_villages = d_i_j.shape[0]
x_i_j = np.empty((no_of_villages, no_of_villages), dtype=pl.LpVariable)
y_i_j = np.empty((no_of_villages, no_of_villages), dtype=pl.LpVariable)

# Decision variables
v = pl.LpVariable("v", lowBound=1, upBound=no_of_villages, cat=pl.LpInteger)

for i in range(no_of_villages):
    for j in range(no_of_villages):
        if i != j:
            y_i_j[i][j] = pl.LpVariable(
                "y_"+str(i+1)+"_"+str(j+1), lowBound=0, upBound=max_distance, cat=pl.LpInteger)
            x_i_j[i][j] = pl.LpVariable(
                "x_"+str(i+1)+"_"+str(j+1), cat=pl.LpBinary)

# Problem
prob = pl.LpProblem("partA", pl.LpMinimize)

# Objective function
prob += v

# Constraints
# 2
prob += pl.lpSum([x_i_j[start_village - 1][i]
                  for i in range(no_of_villages) if i != start_village - 1]) == v
# 3
prob += pl.lpSum([x_i_j[i][start_village - 1]
                  for i in range(no_of_villages) if i != start_village - 1]) == v

for i in range(no_of_villages):

    if i != start_village - 1:
        # 4
        prob += pl.lpSum([x_i_j[j][i]
                          for j in range(no_of_villages) if i != j]) == 1
        # 5
        prob += pl.lpSum([x_i_j[i][j]
                          for j in range(no_of_villages) if i != j]) == 1
        # 12
        prob += pl.lpSum([y_i_j[i][j] for j in range(no_of_villages) if i != j]) - \
            pl.lpSum([y_i_j[j][i] for j in range(no_of_villages) if i != j]) - \
            pl.lpSum([x_i_j[i][j] * d_i_j[i][j]
                      for j in range(no_of_villages) if i != j]) == 0

    for j in range(no_of_villages):
        if i != j:
            # 13
            prob += y_i_j[i][j] <= max_distance * x_i_j[i][j]
            # 15
            prob += y_i_j[i][j] >= 0

    if i != start_village - 1:
        # 14
        prob += y_i_j[start_village - 1][i] == d_i_j[start_village - 1][i] * \
            x_i_j[start_village - 1][i]

# Solve
status = prob.solve(pl.CPLEX_PY(msg=0))

if (status == 1):
    # If there is an optimal solution print the result
    path_list = []
    for v in prob.variables():
        if (v.varValue == 1 and v.name[0] == "x"):
            temp = (v.name[2:]).split('_')
            path_list.append((int(temp[0]), int(temp[1])))

    while len(path_list) != 0:
        print("\nA path is", start_village, end='')
        curr = start_village
        distance = 0
        for src, dest in path_list:
            if src == curr:
                curr = dest
                path_list.remove((src, dest))
                distance += d_i_j[src-1][dest-1]
                break

        while curr != start_village:
            print(" ->", curr, end='')
            for src, dest in path_list:
                if src == curr:
                    curr = dest
                    path_list.remove((src, dest))
                    distance += d_i_j[src-1][dest-1]
                    break

        print(" ->", curr, end='.')
        print("\nTime is", distance / velocity, end='.\n')

    print("\nMinimum number of volunteers is " +
          str(int(prob.objective.value())) + ".")
else:
    print("No optimal solution.")
