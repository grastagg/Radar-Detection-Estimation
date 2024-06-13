import numpy as np

# Step 2: Create a NumPy array with repeated elements
array_with_repeats = np.array([1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7])

# Step 3: Find unique elements and their counts
unique_elements, counts = np.unique(array_with_repeats, return_counts=True)

# Step 4: Filter elements that occur exactly once
unique_elements_single_occurrence = unique_elements[counts == 1]

# Step 5: Get indices of these unique elements in the original array
indices = np.array([index for index, element in enumerate(array_with_repeats) if element in unique_elements_single_occurrence])

print("Original array:", array_with_repeats)
print("Indices of elements that appear only once:", indices)
