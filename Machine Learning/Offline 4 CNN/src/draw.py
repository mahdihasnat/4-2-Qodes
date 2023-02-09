import pandas as pd
from matplotlib import pyplot as plt
str_pre='custom_'
file_name='data/custom.csv'
df = pd.read_csv(file_name, sep=',', header=None)
print(df)
# stat_d = pd.DataFrame(columns=["epoch","loss_validation","loss_train","f1_validation","accuracy_validation","f1_test","accuracy_test"])
x_axis = df[0].to_numpy()[1:].astype(int)
y_loss_validation = df[1].to_numpy()[1:]
y_loss_validation = y_loss_validation.astype(float)
y_loss_train = df[2].to_numpy()[1:].astype(float)
y_f1 = df[3].to_numpy()[1:].astype(float)
y_accuracy = df[4].to_numpy()[1:].astype(float)

print(x_axis)
print(y_loss_validation)

plt.figure()
plt.plot(x_axis,y_loss_validation)
plt.xticks(x_axis)
plt.xlabel("Epoch")
plt.title("Validation Cross entropy Loss")
plt.savefig("figs/"+str_pre+"validation_loss.png")

plt.figure()
plt.plot(x_axis,y_loss_train)
plt.xticks(x_axis)
plt.xlabel("Epoch")
plt.title("Train Cross entropy Loss")
plt.savefig("figs/"+str_pre+"train_loss.png")

plt.figure()
plt.plot(x_axis,y_accuracy)
plt.xticks(x_axis)
plt.xlabel("Epoch")
plt.title("Accuracy")
plt.savefig("figs/"+str_pre+"validation_acc.png")

plt.figure()
plt.plot(x_axis,y_f1)
plt.xticks(x_axis)
plt.xlabel("Epoch")
plt.title("F1 score")
plt.savefig("figs/"+str_pre+"validation_f1.png")