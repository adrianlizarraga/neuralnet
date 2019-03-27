#define NDEBUG
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "MLPNetwork.h"

using namespace std;

int main(int argc, char **argv) {
    srand(time(NULL));
    
    alai::MLPNetwork nn(2, {2, 1});
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> targets;
    
    // Initialize traning set
    for (double i = 0; i <= 1.0; i++) {
        for (double j = 0; j <= 1.0; j++) {
            Eigen::VectorXd in(2);
            Eigen::VectorXd out(1);
            in(0, 0) = i;
            in(1, 0) = j;
            out(0, 0) = ((i+j) == 1.0) ? 1.0 : 0.0;
            inputs.push_back(in);
            targets.push_back(out);
        }
    }

    // Train in a stupid way.
    for (int k = 0; k <= 3000; k++) {
        double errorSum = 0.0;
        
        for (int i = 0; i < inputs.size(); i++) {
            errorSum += nn.train(inputs[i], targets[i]);
        }

        if (k % 500 == 0) {
            cout << "Iteration:\t" << k << "\tError:\t" << (errorSum / (double) inputs.size()) << endl;
        }
    }

    // Print weights
    nn.print();

    // Compute XOR truth table
    for (double i = 0; i <= 1.0; i++) {
        for (double j = 0; j <= 1.0; j++) {
            cout << "\n(" << i << ", " << j << "):" << endl;
            std::vector<double> output = nn.compute(std::vector<double>{i, j});
            cout << output[0] << endl;
        }
    }

}
