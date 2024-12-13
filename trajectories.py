import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV-файла
# Предполагается, что файл называется "trajectories.csv"
# и имеет формат: time, traj_0, traj_1, traj_2, ...
data = pd.read_csv("trajectories.csv")

# Извлекаем время (первый столбец) и остальные траектории
time = data.iloc[:, 0]  # Первая колонка с временем
trajectories = data.iloc[:, 1:]  # Все остальные колонки с траекториями

# Построение графика
plt.figure(figsize=(10, 6))
for col in trajectories.columns:
    plt.plot(time, trajectories[col], label=col)

plt.title("Trajectories of Asset Prices")
plt.xlabel("Time (years)")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.show()
