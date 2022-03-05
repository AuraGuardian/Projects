
import math 
a = []
input_a = 0

# Getting Input in the form of a list [a0, a1, a2, a3 ..... an]

while input_a != 'end':
    
    input_a = input()
    
    if input_a != 'end':
        
        a.append(int(input_a))

print(a)

# Creating the functiion in the form of F(x)
def f(x):
    ans = 0
    for i in range(len(a)):
        ans = ans + a[i] * ((x)**i)
    return ans

# f1 in Sum of list f(1); fans is 
f1 = f(1)
fans = f(f1)
fans2 = f(f1*f1)

# We know fans lies between f1^(n1) to f1^(n1+1)
# We know fans2 lies between f1^(2n2) to f1^(2n2 +1)
n1 = int(math.log(fans)/math.log(f1))
n2 = int(math.log(fans2)/math.log(f1))

print(n1)
print(n2)

if (n2/n1) == 2:
    ans = n1
elif (n2/n1) > 2:
    ans = n1 + 1
elif (n2/n1) < 2:
    ans = n1 - 1
    
print("Order: " + str(ans))

