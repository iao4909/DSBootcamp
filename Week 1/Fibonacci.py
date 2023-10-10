# Question 1

# Display Fibonacci Series upto 10 terms
# Fibonacci sequence is a sequence in which each number is the sum of the two preceding ones
# Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, etc.

num1 = 0
num2 = 1
print(num1)
print(num2)

i = 0
while i<8:
    fibonaccinum = num1 + num2
    print(fibonaccinum)
    num1 = num2
    num2 = fibonaccinum
    i+=1

