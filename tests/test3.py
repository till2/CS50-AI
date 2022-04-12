unassigned_vars = ["A", "B", "C"]

remaining_values = {
    "A": 2,
    "B": 2,
    "C": 2
}

degrees = {
    "A": 1,
    "B": 2,
    "C": 3
}


sorted_unassigned_vars = sorted(unassigned_vars,key=lambda var: (remaining_values[var], -degrees[var]))

print(sorted_unassigned_vars)

# sort by:
# first key: fewest number of remaining values in its domain (ascending)
# second key: degree/ neighbors-count (descending -> same as ascending negated degrees)

"""
first key:
A/C,B

second key:
C,B,A

first key then second key:
C,A,B


"""