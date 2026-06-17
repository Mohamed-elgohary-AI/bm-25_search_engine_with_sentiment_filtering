import time
import numpy as np

def benchmark(pipeline, queries, runs=50):
    latencies = []

    for _ in range(runs):
        for q in queries:
            start = time.perf_counter()
            pipeline.search(q, top_k=10)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    print("P50:", np.percentile(latencies, 50), "ms")
    print("P90:", np.percentile(latencies, 90), "ms")
    print("P95:", np.percentile(latencies, 95), "ms")
    print("P99:", np.percentile(latencies, 99), "ms")