#include "MLPNetwork.h"
#include <cmath>
#include <iostream>

namespace alai {

/////////////////////////////////////////////////////
// Sigmoid activation function
/////////////////////////////////////////////////////
Sigmoid::Sigmoid(double scaling) : scaling(scaling) {}

double Sigmoid::operator()(double x, const Eigen::VectorXd &outputs) { return 1.0 / (1.0 + this->scaling * exp(-x)); }

double Sigmoid::derivative(double fx) { return this->scaling * fx * (1.0 - fx); }

Sigmoid::~Sigmoid() {}

/////////////////////////////////////////////////////
// Hyperbolic tangent activation function
/////////////////////////////////////////////////////
TanH::TanH(double scaling) : scaling(scaling) {}

double TanH::operator()(double x, const Eigen::VectorXd &outputs) { return tanh(this->scaling * x); }

double TanH::derivative(double fx) { return this->scaling - this->scaling * (fx * fx); }

TanH::~TanH() {}

/////////////////////////////////////////////////////
// Softmax activation function
/////////////////////////////////////////////////////
Softmax::Softmax(double scaling) : scaling(scaling) {}

double Softmax::operator()(double x, const Eigen::VectorXd &outputs) {
    double sum = outputs.unaryExpr([](double elem) { return exp(elem); }).sum();

    return exp(this->scaling * x) / sum;
}

double Softmax::derivative(double fx) { return this->scaling * fx * (1.0 - fx); }

Softmax::~Softmax() {}

/////////////////////////////////////////////////////
// Perceptron layer
/////////////////////////////////////////////////////
Layer::Layer(int nodes) : Layer(nodes, std::make_shared<Sigmoid>()) {}

Layer::Layer(int nodes, PActivationFunction activationFunction) : Layer(nodes, activationFunction, RANDOM) {}

Layer::Layer(int nodes, PActivationFunction activationFunction, WeightInitializer weightInitializer)
    : nodes(nodes), weightInitializer(weightInitializer), activationFunction(activationFunction) {}

void Layer::initWeights(int weightsPerNode) {
    switch (this->weightInitializer) {
    case RANDOM:
        this->weights = Eigen::MatrixXd::Random(this->nodes, weightsPerNode);
        this->biases = Eigen::VectorXd::Random(this->nodes);
        break;
    case ONES:
        this->weights = Eigen::MatrixXd::Ones(this->nodes, weightsPerNode);
        this->biases = Eigen::VectorXd::Ones(this->nodes);
        break;
    case ZEROS:
        this->weights = Eigen::MatrixXd::Zero(this->nodes, weightsPerNode);
        this->biases = Eigen::VectorXd::Zero(this->nodes);
        break;
    }
}

void Layer::compute(const Eigen::VectorXd &input) {
    this->inputs = input;

    Eigen::MatrixXd layerWeightedSum = (this->weights * this->inputs) + this->biases;
    this->outputs = layerWeightedSum.unaryExpr([this, &layerWeightedSum](double elem) {
        return (*this->activationFunction)(elem, layerWeightedSum); 
    });
}

void Layer::backpropagate(const Eigen::VectorXd &prevError, const Eigen::MatrixXd &prevWeights) {
    // Compute: hadamardProduct(dot(prevWeights, prevError), activationDerivatives)
    Eigen::VectorXd activationDerivatives =
        this->outputs.unaryExpr([this](double elem) { return this->activationFunction->derivative(elem); });

    // Save this layer's errors to update weights later AND for the next layer
    // to use in back propagation.
    this->deltas = (prevWeights.transpose() * prevError).cwiseProduct(activationDerivatives);
}

void Layer::update(double learningRate) {
    this->weights += this->deltas * this->inputs.transpose() * learningRate;
    this->biases += this->deltas * learningRate;
}

/////////////////////////////////////////////////////
// Multi-layer perceptron
/////////////////////////////////////////////////////
MLPNetwork::MLPNetwork(int inputs, const std::vector<Layer> &layers) : inputs(inputs), layers(layers) {
    int numInputs = inputs;

    for (int i = 0; i < this->layers.size(); i++) {
        this->layers[i].initWeights(numInputs);
        numInputs = this->layers[i].nodes;
    }
}

std::vector<double> MLPNetwork::compute(const std::vector<double> &input) {
    const Eigen::VectorXd inputVector = Eigen::Map<const Eigen::VectorXd>(input.data(), input.size());
    Eigen::VectorXd outputVector = this->compute(inputVector);

    return std::vector<double>(outputVector.data(), outputVector.data() + outputVector.size());
}

Eigen::MatrixXd MLPNetwork::compute(const Eigen::VectorXd &input) {
    int numLayers = this->layers.size();

    Eigen::VectorXd inputs = input;
    for (int i = 0; i < numLayers; i++) {
        this->layers[i].compute(inputs);
        inputs = this->layers[i].outputs;
    }

    return this->layers.back().outputs;
}

double MLPNetwork::train(const std::vector<double> &input, const std::vector<double> &target, double learningRate) {
    const Eigen::VectorXd inputVector = Eigen::Map<const Eigen::VectorXd>(input.data(), input.size());
    const Eigen::VectorXd targetVector = Eigen::Map<const Eigen::VectorXd>(target.data(), target.size());

    return this->train(inputVector, targetVector, learningRate);
}

double MLPNetwork::train(const Eigen::VectorXd &input, const Eigen::VectorXd &target, double learningRate) {
    this->compute(input);

    int numLayers = this->layers.size();
    Eigen::MatrixXd prevErrors = target - this->layers.back().outputs;
    Eigen::MatrixXd prevWeights = Eigen::MatrixXd::Identity(this->layers.back().nodes, this->layers.back().nodes).transpose();
    double error = 0.5 * prevErrors.unaryExpr([](double elem) { return elem * elem; }).sum();

    // Start back propagation at the output layer.
    this->layers.back().backpropagate(prevErrors, prevWeights);

    // Backwards propagation of errors in the hidden layers.
    for (int i = numLayers - 2; i >= 0; i--) {
        prevErrors = this->layers[i + 1].deltas;
        prevWeights = this->layers[i + 1].weights;

        this->layers[i].backpropagate(prevErrors, prevWeights);
    }

    // Update all weights/biases.
    for (int i = 0; i < numLayers; i++) {
        this->layers[i].update(learningRate);
    }

    return error;
}

double MLPNetwork::weightAt(int layer, int node, int input) const { return this->layers[layer].weights(node, input); }

double MLPNetwork::biasAt(int layer, int node) const { return this->layers[layer].biases(node, 0); }

void MLPNetwork::print() const {
    using namespace std;

    cout << endl;
    for (int i = 0; i < this->layers.size(); i++) {
        cout << "Layer " << i << " weights: " << endl;
        cout << this->layers[i].weights << endl;
        cout << "Layer " << i << " biases: " << endl;
        cout << this->layers[i].biases << endl << endl;
    }
    cout << endl;
}
} // namespace alai
