import matplotlib.pyplot as plt
import numpy as np

weightf = "weights_run_5.txt"
weightf2 = "weights_run_7.txt"
weightf3 = "weights_run_13.txt"
epochs = []
train_loss = []
test_loss = []
times=[]

times2 = []
epochs2 = []
train_loss2 = []
test_loss2 = []

epochs3 = []
train_loss3 = []
test_loss3 = []
times3 = []

with open(weightf, "r") as f:
    for line in f:
        if "Epoch" in line:
            parts = line.split(", ")
            epoch = int(parts[0].split()[1])
            train_loss_value = float(parts[1].split()[2])
            test_loss_value = float(parts[2].split()[2])
            time_value = float(parts[3].split()[1][:-1])

            epochs.append(epoch)
            train_loss.append(train_loss_value)
            test_loss.append(test_loss_value)
            times.append(time_value)

with open(weightf2, "r") as f:
    for line in f:
        if "Epoch" in line:
            parts = line.split(", ")
            epoch = int(parts[0].split()[1])
            train_loss_value = float(parts[1].split()[2])
            test_loss_value = float(parts[2].split()[2])
            time_value = float(parts[3].split()[1][:-1])

            epochs2.append(epoch)
            train_loss2.append(train_loss_value)
            test_loss2.append(test_loss_value)
            times2.append(time_value)

with open(weightf3, "r") as f:
    for line in f:
        if "Epoch" in line:
            parts = line.split(", ")
            epoch = int(parts[0].split()[1])
            train_loss_value = float(parts[1].split()[2])
            test_loss_value = float(parts[2].split()[2])
            time_value = float(parts[3].split()[1][:-1])

            epochs3.append(epoch)
            train_loss3.append(train_loss_value)
            test_loss3.append(test_loss_value)
            times3.append(time_value)

plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1) 
plt.plot(epochs, train_loss, label="GD Train Loss")
plt.plot(epochs, test_loss, label="GD Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(2, 3, 2)
plt.plot(np.cumsum(times), train_loss, label="GD Train Loss")
plt.plot(np.cumsum(times), test_loss, label="GD Test Loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(2, 3, 3)
plt.plot(epochs2, train_loss2, label="SGD Train Loss")
plt.plot(epochs2, test_loss2, label="SGD Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(2, 3, 4)
plt.plot(np.cumsum(times2), train_loss2, label="SGD Train Loss")
plt.plot(np.cumsum(times2), test_loss2, label="SGD Test Loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(2, 3, 5) 
plt.plot(epochs3, train_loss3, label="ADAM Train Loss")
plt.plot(epochs3, test_loss3, label="ADAM Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(2, 3, 6) 
plt.plot(np.cumsum(times3), train_loss3, label="ADAM Train Loss")
plt.plot(np.cumsum(times3), test_loss3, label="ADAM Test Loss")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()
