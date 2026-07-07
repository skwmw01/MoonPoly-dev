import csv

csv_file = 'results.csv'

# M, N, K = 105, 32768, 512
# M, N, K = 105, 1024, 12544
# M, N, K = 4096, 4096, 4096
# M, N, K = 213, 4096, 14336
# M, N, K = 213, 4096, 14336
M, N, K = 4096, 63, 4096

a100_core = 312.0
a100_mem = 1935.0

point = a100_core / (a100_mem / 1024.0)

print(f"M: {M}, N: {N}, K: {K}")


flops = 2 * M * N * K
print(f"FLOPs: {flops}")

#FP16
memory_accesses = M * K + K * N + M * N
memory_accesses = memory_accesses * 2
print(f"Memory Accesses: {memory_accesses}")
# Arithmetic Intensity
ai = flops / memory_accesses
print(f"Arithmetic Intensity: {ai}")
print ("point", point)

# Seconds
time = 0.0000307616
print(f"Time: {time}")
tflps = flops / time / 1e12
print(f"TFLOPS: {tflps}")
bw = memory_accesses / time / 1e9
print(f"Bandwith:{bw}")

# Write to CSV
# with open(csv_file, 'a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([M, N, K, time, flops, memory_accesses, ai, tflps])