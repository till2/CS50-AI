import numpy as np

X1 = [0,1]
X2 = [0,1]
X3 = [0,1]

print("\nNeural network structure:")
print("\n     x1")
print("       \\")
print("      (*w1)    +w0")
print("         \\     |")
print("x2-(*w2)- + -> y -> step-function -> out")
print("         /")
print("      (*w3)")
print("       /")
print("     x3")

print("\n\n\nTruth table:\n")

print("x1 | x2 | x3 |  y | out")
print("-----------------------")

for x1 in X1:
    for x2 in X2:
        for x3 in X3:

            print(f" {x1} |  {x2} |  {x3} | ", end = "")

            W = [-3, 1, 1, 1] # w0 (bias), w1, w2
            X = [1, x1, x2, x3]

            # dot product
            y = np.dot(W, X)

            # activation : step function
            out = 1 if y >= 0 else 0

            print(f"{y:2d} | {out:2d}")

print("\nout = AND(x1, x2, x3)\n")