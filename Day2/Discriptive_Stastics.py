


import numpy as np
import pandas as pd
from statistics import multimode

# ---------------TOPIC 1---------------
# Mean, Median, Mode, Range


sales = [100, 120, 120, 130, 150, 180, 400]
print("---------------TOPIC 1---------------")

print("Mean:", np.mean(sales))
print("Median:", np.median(sales))
print("Mode:", multimode(sales))
print("Range:", np.max(sales) - np.min(sales))

df = pd.DataFrame({"sales": sales})
print("\nSummary statistics:")
print(df["sales"].describe())


# ---------------TOPIC 2---------------

# Variance and Standard Deviation

sales = [2, 4, 6]

print("---------------TOPIC 2---------------")
print("Mean:", np.mean(sales))

# Population calculations
print("Population variance:", np.var(sales))
print("Population standard deviation:", np.std(sales))

# Sample calculations
print("Sample variance:", np.var(sales, ddof=1))
print("Sample standard deviation:", np.std(sales, ddof=1))

# Pandas uses sample variance and standard deviation by default
df = pd.DataFrame({"sales": sales})
print("Pandas variance:", df["sales"].var())
print("Pandas standard deviation:", df["sales"].std())




# ---------------TOPIC 3---------------
#  Percentaile, Outliner, Quartlies

sales = [100, 110, 105, 115, 108, 112, 900]

q1 = np.percentile(sales, 25)
median = np.percentile(sales, 50)
q3 = np.percentile(sales, 75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = [
    value for value in sales
    if value < lower_bound or value > upper_bound
]
print("---------------TOPIC 3---------------")
print("Q1:", q1)
print("Median:", median)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower boundary:", lower_bound)
print("Upper boundary:", upper_bound)
print("Potential outliers:", outliers)

df = pd.DataFrame({"sales": sales})
print("\nPercentiles:")
print(df["sales"].quantile([0.25, 0.50, 0.75]))


# ---------------TOPIC 4---------------
# Distribution, Skewness and Histograms.

import matplotlib.pyplot as plt

sales = [10, 12, 14, 15, 18, 21, 22, 25, 28, 30]

print("---------------TOPIC 4---------------")


plt.hist(sales, bins=[10, 20, 30, 40], edgecolor="black")

plt.title("Distribution of Daily Sales")
plt.xlabel("Sales Range")
plt.ylabel("Number of Days")
plt.show()
