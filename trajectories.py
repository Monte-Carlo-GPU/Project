import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("trajectories.csv")

time = data.iloc[:, 0]  
trajectories = data.iloc[:, 1:]  

plt.figure(figsize=(10, 6))
for col in trajectories.columns:
    plt.plot(time, trajectories[col], label=col)

plt.title("Trajectories of Asset Prices")
plt.xlabel("Time (years)")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.show()
