# Question 2
# Display numbers at the odd indices of a list

list = [1,2,3,4,5,6,7,8,9,10]

i = 0

while i >= 0:
    if i % 2 ==1:
        print(list[i])
        i+=1
    else:
        i+=1