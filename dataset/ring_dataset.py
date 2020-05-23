from sklearn.datasets import make_circles

N_SAMPLES = 10000

ring_inputs, ring_labels = make_circles(n_samples=N_SAMPLES, noise=0.15, factor=0.2)
