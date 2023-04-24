import math

def binary(num):
    x=int(math.log2(num))+1
    for i in range(x):
        num=(num ^ (1<<i))
        print(num)
for i in range(int(input())):
    n=int(input())
    binary(n)
