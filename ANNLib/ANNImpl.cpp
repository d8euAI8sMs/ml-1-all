#define ANNDLL_EXPORTS

#include <iostream>

#include <ANN.h>

namespace ANN
{

    class BackPropagationNetwork : public NeuralNetwork
    {

    public:

        BackPropagationNetwork(
            std::vector<int> & configuration,
	        NeuralNetwork::ActivationType activation_type)
        {
            this->configuration = configuration;
            this->activation_type = activation_type;
            this->scale = 1;
        }

    public:

        ANNDLL_API virtual std::string GetType();
        ANNDLL_API virtual std::vector<float> Predict(std::vector<float> & input);
        ANNDLL_API virtual float MakeTrain(
            std::vector<std::vector<float>> & inputs,
            std::vector<std::vector<float>> & outputs,
            int max_iters = 10000,
            float eps = 0.1,
            float speed = 0.1,
            bool std_dump = false
        );
        
    private:

        /**
         * Performs forward data propagation and fills
         * the `per_layer_output` with per-layer output data.
         * 
         * @param input            The network input data
         * @param per_layer_output The per-layer output data
         */
        void ForwardPropagate(std::vector < float > & input,
                              std::vector < std::vector < float > > & per_layer_output);

        /**
         * Performs backward error propagation and weight
         * matrix correction.
         * 
         * @param input            The network input
         * @param sigma            The error obtained from the output layer
         * @param per_layer_output The per-layer output data
         * @param speed            The training speed parameter
         */
        void BackwardPropagate(std::vector<float> & input,
                               std::vector<float> & sigma,
                               std::vector < std::vector < float > > & per_layer_output,
                               float speed);
    };
}

std::shared_ptr<ANN::NeuralNetwork> ANN::CreateNeuralNetwork(
    std::vector<int> & configuration,
    NeuralNetwork::ActivationType activation_type)
{
    return std::make_shared<ANN::BackPropagationNetwork>(configuration, activation_type);
}

std::string ANN::BackPropagationNetwork::GetType()
{
    return "Back Propagation Network by Vasilevsky Alexander";
}

std::vector<float> ANN::BackPropagationNetwork::Predict(std::vector<float> & input)
{
    std::vector < float > in = input;
    std::vector < float > out;
    for (size_t i = 0; i < weights.size(); ++i) // layers
    {
        out.resize(weights[i].size());
        for (size_t j = 0; j < weights[i].size(); ++j) // neurons in i-th layer
        {
            float neuron_input = 0;
            for (size_t k = 0; k < in.size(); ++k) // (i-1)-th layer output
            {
                neuron_input += in[k] * weights[i][j][k];
            }
            out[j] = Activation(neuron_input);
        }
        in.swap(out);
    }
    return in;
}

void ANN::BackPropagationNetwork::ForwardPropagate(std::vector<float> & input,
                                                   std::vector < std::vector < float > > & per_layer_output)
{
    std::vector < float > in = input;
    std::vector < float > out;

    for (size_t i = 0; i < weights.size(); ++i) // layers
    {
        out.resize(weights[i].size());
        per_layer_output[i].resize(weights[i].size());
        for (size_t j = 0; j < weights[i].size(); ++j) // neurons in i-th layer
        {
            float neuron_input = 0;
            for (size_t k = 0; k < in.size(); ++k) // (i-1)-th layer output
            {
                neuron_input += in[k] * weights[i][j][k];
            }
            per_layer_output[i][j] = Activation(neuron_input);
            out[j] = per_layer_output[i][j];
        }
        in.swap(out);
    }
}

void ANN::BackPropagationNetwork::BackwardPropagate(std::vector<float> & input,
                                                    std::vector<float> & sigma,
                                                    std::vector < std::vector < float > > & per_layer_output,
                                                    float speed)
{
    std::vector < float > in = sigma;
    std::vector < float > out;

    for (size_t i = weights.size() - 1; i-- > 0;) // layers
    {
        out.resize(weights[i].size());
        for (size_t j = 0; j < weights[i].size(); ++j) // neurons in i-th layer
        {
            float neuron_input = 0;
            for (size_t k = 0; k < in.size(); ++k) // (i+1)-th layer output
            {
                neuron_input += in[k] * weights[i + 1][k][j];
            }
            neuron_input *= ActivationDerivative(per_layer_output[i][j]);
            out[j] = neuron_input;
        }
        for (size_t k = 0; k < in.size(); ++k) // (i+1)-th layer output
        {
            for (size_t j = 0; j < weights[i].size(); ++j) // neurons in i-th layer
            {
                weights[i + 1][k][j] += speed * per_layer_output[i][j] * in[k];
            }
        }
        in.swap(out);
    }
    for (size_t k = 0; k < in.size(); ++k) // 1st hidden layer output
    {
        for (size_t j = 0; j < input.size(); ++j) // neurons in input layer
        {
            weights[0][k][j] += speed * input[j] * in[k];
        }
    }
}

float ANN::BackPropagationNetwork::MakeTrain
(
    std::vector < std::vector < float > > & inputs,
    std::vector < std::vector < float > > & outputs,
    int max_iters,
    float eps,
    float speed,
    bool std_dump
)
{
    weights.resize(configuration.size() - 1);
    for (size_t i = 0; i < weights.size(); ++i) // layers
    {
        weights[i].resize(configuration[i + 1]);
        for (size_t j = 0; j < weights[i].size(); ++j) // neurons
        {
            weights[i][j].resize(configuration[i]);
            for (size_t k = 0; k < weights[i][j].size(); ++k)
            {
                weights[i][j][k] = (float)rand() / RAND_MAX;
            }
        }
    }

    float err;
    int iterations = 0;

    do
    {
        err = 0;
        for (size_t e = 0; e < inputs.size(); ++e) // for all data set
        {
            std::vector < std::vector < float > > layer_outputs(weights.size());

            ForwardPropagate(inputs[e], layer_outputs);

            std::vector < float > & output = layer_outputs.back();

            std::vector < float > sigma(output.size());

            for (size_t el = 0; el < output.size(); ++el)
            {
                float diff = outputs[e][el] - output[el];
                err += diff * diff;
                sigma[el] = diff * ActivationDerivative(output[el]);
            }

            BackwardPropagate(inputs[e], sigma, layer_outputs, speed);
        }
        if (std_dump && ((iterations % (max_iters / 100)) == 0))
        {
            std::cout << iterations << ": " << err << std::endl;
        }
        ++iterations;
    } while ((err > eps) && (iterations < max_iters));

    is_trained = true;
    
    if (std_dump)
    {
        std::cout << iterations << ": " << err << std::endl;
    }

    return err;
}
