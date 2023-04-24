t=int(input())
for i in range(t):
    n=int(input())
    for j in range(n):
        for k in range(j):
            if k==0 or j==k:
                print(1,end=" ")
            else:
                print(0,end=" ")
        print()
