import numpy as np
import concurrent.futures
import csv

# Function to perform addition on the n-th element


def add_elements(params):
    list1_value, list2_value = params
    return [list1_value, list2_value, list1_value + list2_value]


# Example lists
list1 = [1, 2, 3, 4, 5]
list2 = [10, 20, 30, 40, 50]

# Combine the lists into a list of tuples to pass as parameters
params = [(list1[i], list2[i]) for i in range(len(list1))]

# Initialize results array
results = np.zeros((len(list1), 3))  # 3 columns: list1 value, list2 value, sum

# Number of threads to run in parallel (e.g., 5 cores)
num_workers = 5  # You can set this to the number of CPU cores you have

# Run the additions in parallel using threads
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(add_elements, param) for param in params]
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        results[i, :] = future.result()

# Optionally, save results to CSV
with open('addition_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['List1 Value', 'List2 Value', 'Sum'])  # Header
    writer.writerows(results)

print(results)
