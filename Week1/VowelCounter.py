# Question 5 
# Write a function that takes a word as an argument and returns the number of vowels in the word

word = input("Please input a word: ")

vowelcount = 0
vowels = ['a','e','i','o','u']

for character in word:
    if character in vowels:
        vowelcount +=1 
        
print(vowelcount)
