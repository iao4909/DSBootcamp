# Question 6
# Iterate through the following list of animals and print each one in all caps.

  

animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']

i=0 

while i <= len(animals) - 1:
    word = animals[i]
    print(word.upper())
    i+=1
