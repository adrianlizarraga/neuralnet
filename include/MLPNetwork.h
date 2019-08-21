#ifndef AL_AI_MLP_NETWORK_H
#define AL_AI_MLP_NETWORK_H
#include "eigen3/Eigen/Dense"
#include <memory>
#include <vector>
#include <random>

namespace alai {

/////////////////////////////////////////////////////
// Activation functions
/////////////////////////////////////////////////////
class ActivationFunction {
  public:
    virtual double operator()(double x, const Eigen::VectorXd &outputs) = 0;
    virtual double derivative(double fx) = 0;
    virtual ~ActivationFunction() {}
};

class Sigmoid : public ActivationFunction {
  private:
    double scaling;

  public:
    Sigmoid(double scaling = 1.0);
    virtual ~Sigmoid();
    virtual double operator()(double x, const Eigen::VectorXd &outputs);
    virtual double derivative(double fx);
};

class TanH : public ActivationFunction {
  private:
    double scaling;

  public:
    TanH(double scaling = 1.0);
    virtual ~TanH();
    virtual double operator()(double x, const Eigen::VectorXd &outputs);
    virtual double derivative(double fx);
};

class Softmax : public ActivationFunction {
  private:
    double scaling;

  public:
    Softmax(double scaling = 1.0);
    virtual ~Softmax();
    virtual double operator()(double x, const Eigen::VectorXd &outputs);
    virtual double derivative(double fx);
};

typedef std::shared_ptr<ActivationFunction> PActivationFunction;

/////////////////////////////////////////////////////
// Perceptron layer
/////////////////////////////////////////////////////
class Layer {
  public:
    enum WeightInitializer { RANDOM, ONES, ZEROS };

    Layer(int nodes);
    Layer(int nodes, PActivationFunction activationFunction);
    Layer(int nodes, PActivationFunction activationFunction, WeightInitializer weightInitializer);

    friend class MLPNetwork;

  private:
    void initWeights(int weightsPerNode);
    void randomizeWeights();
    void compute(const Eigen::VectorXd &input);
    void backpropagate(const Eigen::VectorXd &prevError, const Eigen::MatrixXd &prevWeights);
    void update(double learningRate);
    void mutate(double probability);

    int nodes;
    PActivationFunction activationFunction;
    WeightInitializer weightInitializer;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd inputs;
    Eigen::VectorXd outputs;
    Eigen::MatrixXd deltas;
    std::default_random_engine engine;
};

/////////////////////////////////////////////////////
// Multi-layer perceptron
/////////////////////////////////////////////////////
class MLPNetwork {
  public:
    MLPNetwork(int inputs, const std::vector<Layer> &layers);

    std::vector<double> compute(const std::vector<double> &input);
    Eigen::MatrixXd compute(const Eigen::VectorXd &input);

    double train(const std::vector<double> &input, const std::vector<double> &target, double learningRate = 0.55);
    double train(const Eigen::VectorXd &input, const Eigen::VectorXd &target, double learningRate = 0.55);

    double weightAt(int layer, int node, int input) const;
    double biasAt(int layer, int node) const;

    void print() const;
    void randomizeWeights();
    void mutate(double probability);

  private:
    int inputs;
    std::vector<Layer> layers;
};

} // namespace alai
#endif