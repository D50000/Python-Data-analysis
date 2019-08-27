# List of numbers
numbers = range(0, 10) 

# range() function does not store all the values in memory, it would be inefficient.
# To force this function to output all the items, we can use the function list().
print(list(numbers))

# variable to store the sum
sum = 0

# iterate over the list
for num in numbers:
	sum = sum + num

# Output: The sum is 45
print("The sum is", sum)