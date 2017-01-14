#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import mat4py
import numpy

def evalPerceptron(negExs, posExs, w):
    negFails = []
    posFails = []
    for negEx in negExs:
        result = numpy.dot(numpy.transpose(negEx), w).tolist()[0]
        if result >= 0:
            negFails.append(negEx)
    for posEx in posExs:
        result = numpy.dot(numpy.transpose(posEx), w).tolist()[0]
        if result < 0:
            posFails.append(posEx)
    return (negFails, posFails)

def recalculateWeights(w, negFails, posFails):
    w = numpy.transpose(w)
    for negFail in negFails:
        w = numpy.subtract(w, negFail).tolist()
    for posFail in posFails:
        w = numpy.add(w, posFail).tolist()
    w = numpy.transpose(w)
    return w.tolist()

def learn(data, iter):
    w = data['w_init']
    for i in range(0, iter):
        (negFails, posFails) = evalPerceptron(data['neg_examples'], data['pos_examples'], w)
        errors = len(negFails) + len(posFails)
        print("Iteration {}:".format(i))
        print("\tNumber of errors: {}".format(errors))
        print("\tWeights: {}".format(json.dumps(w)))
        if errors == 0:
            return
        w = recalculateWeights(w, negFails, posFails)

def addBias(inputs):
    inputsWithBias = []
    for input in inputs:
        input.append(1)
        inputsWithBias.append(input)
    return inputsWithBias

def load(data_file):
    data = mat4py.loadmat(data_file)
    data['neg_examples'] = addBias(data['neg_examples_nobias'])
    data['pos_examples'] = addBias(data['pos_examples_nobias'])
    return data

def main():
    parser = argparse.ArgumentParser(description="Use a perceptron model to classify two classes")
    parser.add_argument('-d', '--data-file', dest='data_file', required=True, help='The .mat file to provide the input data')
    parser.add_argument('-l', '--iteration-limit', dest='iteration_limit', required=True, help='The max number of iterations to perform before failing')

    args = parser.parse_args()
    learn(load(args.data_file), int(args.iteration_limit))

if __name__ == "__main__":
    main()
