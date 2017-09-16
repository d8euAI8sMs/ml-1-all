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
