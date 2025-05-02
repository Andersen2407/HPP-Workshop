import matplotlib.pyplot as plt

# Replace with your actual timings
process_counts = [5, 10, 15, 20]  # Number of processes used
# Example execution times for each process count (in seconds)
execution_times = [0.0048, 0.0072, 0.0059, 0.0069]

speedup = [execution_times[0] / t for t in execution_times]

plt.figure(figsize=(8, 6))
plt.plot(process_counts, speedup, marker='o', label='Measured Speedup')
plt.plot(process_counts, process_counts, linestyle='--', label='Ideal Linear Speedup')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.title('MPI BFS Speedup')
plt.legend()
plt.grid(True)
#plt.savefig("bfs_speedup.png")
plt.show()