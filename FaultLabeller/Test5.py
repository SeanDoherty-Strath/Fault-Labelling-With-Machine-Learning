

accuracyArray = [0, 0, 0, 0, 0, 0]
accuracyArray = [accuracyArray, accuracyArray, accuracyArray,
                 accuracyArray, accuracyArray, accuracyArray]
activationFunctions = ['relu', 'sigmoid', 'tanh', 'elu', 'linear', 'softmax']
# activationFunctions = ['relu', 'sigmoid']

for i in range(len(activationFunctions)):
    for j in range(len(activationFunctions)):
        accuracy = 0
        for k in range(5):
            accuracy += 1
        accuracy /= 5
        accuracyArray[i][j] = accuracy

print(accuracyArray)
