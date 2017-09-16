#define ANNDLL_EXPORTS

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
