#include "MLPNetwork.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using namespace std;

// The line dividing the two classes.
double func(double x) { return 2 * x + 0.3; }

int main(int argc, char **argv) {
    srand(time(NULL));

    // Network with 1 input, 1 perceptron in the hidden layer, and 2 output perceptrons w/ softmax activation.
    alai::MLPNetwork nn(1, {1, alai::Layer(2, std::make_shared<alai::Softmax>())});
    
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> targets;

    // Initialize traning set
    for (double x = -1.0; x <= 1.0; x += 0.1) {
        Eigen::VectorXd in(1);
        Eigen::VectorXd out(2);
        double f = func(x);
        
        in(0, 0) = x;
        out(0, 0) = f >= 0.0 ? 1.0 : 0.0;
        out(1, 0) = f < 0.0 ? 1.0 : 0.0;
        
        inputs.push_back(in);
        targets.push_back(out);
    }

    // Train in a stupid way.
    for (int k = 0; k <= 500; k++) {
        double errorSum = 0.0;

        for (int i = 0; i < inputs.size(); i++) {
            errorSum += nn.train(inputs[i], targets[i]);
        }

        if (k % 500 == 0) {
            cout << "Iteration:\t" << k << "\tError:\t" << (errorSum / (double)inputs.size()) << endl;
        }
    }

    // Print weights
    nn.print();

    // Run it forward and see what happens
    for (double x = -1.0; x <= 1.0; x += 0.1) {
        std::vector<double> output = nn.compute(std::vector<double>{x});
        double f = func(x);
        double actualClass = f >= 0.0 ? 1.0 : 0.0;
        double predictedClass = output[0] > output[1] ? 1.0 : 0.0;
        bool correct = actualClass == predictedClass;

        cout << x << " => " << (int) actualClass << " - " << (correct ? "Correct" : "Wrong") << endl;

        if (!correct) {
            cout << "\traw output: [" << output[0] << " " << output[1] << "]" << endl;
        }
    }
}
