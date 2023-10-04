import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_dataset.csv", usecols=[1, 2])

plt.scatter(data.YearsExperience, data.Salary)


# mean squared error
# uses the whole dataset!
def loss_function(m, b, points):
    total_error = 0
    # add individual errors squared
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


# batch gradient descent
# uses the whole dataset to compute the m
def batch_gradient_descent(m_now, b_now, points, lr):
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * lr
    b = b_now - b_gradient * lr

    return m, b


# m and b and be anything
m = 0
b = 0
lr = 0.001
epochs = 1000

for i in range(epochs):
    m, b = batch_gradient_descent(m, b, data, lr)
    print(f"epoch {i}: \t m: {m} \t b: {b}")


plt.scatter(data.YearsExperience, data.Salary, color="black")
# plots the line from regression
plt.plot(list(range(1, 11)), [m * x + b for x in range(1, 11)], color="red")
plt.title(f"Batch Gradient Descent {epochs} Epochs, {lr} Learning Rate")
plt.show()
