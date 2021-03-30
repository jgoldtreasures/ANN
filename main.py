import numpy as np


class Node:
    def __init__(self, bias, weights, ins=0):
        self.bias = bias
        self.weights = weights
        self.output = 0
        self.ins = ins


def forward_pass(layer, bias, ins):
    I = np.zeros(len(layer[0]))
    for j in range(len(layer[0])):
        for i in range(len(layer)):
            I[j] += layer[i][j] * ins[i]
        I[j] += bias[j]
    return sigmoid(I)


def error(out, err, w):
    e = []
    for j in range(len(out)):
        val = 0
        for k in range(len(err)):
            val += err[k] * w[j][k]
        val *= (out[j] * (1 - out[j]))
        e.append(val)
    return e


def error_last(out, c):
    e = []
    for j in range(len(out)):
        e.append(out[j]*(1-out[j])*(c[j]-out[j]))
    return e


def update_weights(w, err, out, l):
    for i in range(len(w)):
        for j in range(len(w[0])):
            w[i][j] += l * err[j] * out[i]
    return w


def update_bias(bias, err, l):
    for j in range(len(bias)):
        bias[j] += l * err[j]
    return bias


def sigmoid(output):
    return 1/(1+np.exp(-1*output))


def main():
    l = 0.1

    out1 = [0.6, 0.1]

    weight1 = [[ 0.1,   0,  0.3],
                [-0.2, 0.2, -0.4]]
    weight2 = [[-0.4, 0.2],
               [0.1, -0.1],
               [0.6, -0.2]]
    bias2 = [0.1, 0.2, 0.5]
    bias3 = [-0.1, 0.6]

    out2 = forward_pass(weight1, bias2, out1)
    out3 = forward_pass(weight2, bias3, out2)

    e3 = error_last(out3, [1, 0])
    e2 = error(out2, e3, weight2)

    # weight1 = update_weights(weight1, e2, out1, 0.1)
    weight2 = update_weights(weight2, e3, out2, l)
    weight1 = update_weights(weight1, e2, out1, l)

    bias2 = update_bias(bias2, e2, l)
    bias3 = update_bias(bias3, e3, l)

    #second pass
    out1 = [0.2, 0.3]
    out2 = forward_pass(weight1, bias2, out1)
    out3 = forward_pass(weight2, bias3, out2)

    e3 = error_last(out3, [0, 1])
    e2 = error(out2, e3, weight2)

    # weight1 = update_weights(weight1, e2, out1, 0.1)
    weight2 = update_weights(weight2, e3, out2, l)
    weight1 = update_weights(weight1, e2, out1, l)

    bias2 = update_bias(bias2, e2, l)
    bias3 = update_bias(bias3, e3, l)


main()