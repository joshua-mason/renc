import numpy as np
import matplotlib.pyplot as plt

# Poisson distribution is used to model the number of times an event happens
# in a fixed interval of time/space, given an average rate (lambda).

# Set the average rate (lambda) parameter:
lam = 10  # Expected number of events per interval

# Generate sample data from a Poisson distribution:
samples = np.random.poisson(lam=lam, size=1000)

# Plot histogram of the samples:
plt.hist(
    samples,
    bins=np.arange(0, 15) - 0.5,
    density=True,
    alpha=0.7,
    color="skyblue",
    edgecolor="black",
    label="Samples (simulated)",
)
from scipy.special import factorial

# Overlay the theoretical Poisson PMF
x = np.arange(0, 15)
pmf = (
    np.exp(-lam) * lam**x / factorial(1)
)  # This is wrong for a vector. We'll use scipy:
from scipy.stats import poisson

plt.plot(
    x,
    poisson.pmf(x, lam),
    marker="o",
    linestyle="-",
    color="darkred",
    label="Poisson PMF (theory)",
)

# Add axis labels and a title:
plt.xlabel("Number of events (k)")
plt.ylabel("Probability")
plt.title(f"Poisson Distribution (lambda = {lam})")
plt.legend()
plt.tight_layout()
plt.show()

# Key points:
# - lam: mean number of events per interval (higher lam => distribution shifts right)
# - Poisson is discrete and only defined on non-negative integers
# - Used for counts, e.g. # emails per hour, # arrivals per minute, etc.
