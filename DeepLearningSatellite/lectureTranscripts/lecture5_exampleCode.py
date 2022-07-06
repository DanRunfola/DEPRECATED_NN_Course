maxIterations = 1000
count = 0

while count < maxIterations:
    count = count + 1
    W_gradient_dW = calculateGradient(lossFunction, X, W)
    W = W + -1 * (stepSize * W_gradient_dW)


while count < maxIterations:
    count = count + 1
    X_sample = X.sample(n=256)
    W_gradient_dW = calculateGradient(lossFunction, X_sample, W)
    W = W + -1 * (stepSize * W_gradient_dW)

    

