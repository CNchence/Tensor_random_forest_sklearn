add1 = []
for i in range(24*24 - 1):
    for j in range(i + 1, 24*24, 1):
        add1.append((i,j))
print(add1[:600])