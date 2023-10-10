# Question 3
# Your task is to count the number of different words in this text


examplestring = """
	ChatGPT has created this text to provide tips on creating interesting paragraphs. 
	First, start with a clear topic sentence that introduces the main idea. 
	Then, support the topic sentence with specific details, examples, and evidence.
	Vary the sentence length and structure to keep the reader engaged.
	Finally, end with a strong concluding sentence that summarizes the main points.
	Remember, practice makes perfect!
	"""

words = examplestring.split(" ")
# print(words[1])
wordcount = len(words)
print(wordcount)
