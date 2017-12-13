#define ANNDLL_EXPORTS
#include <ANN.h>
#include <fstream>
#include <iomanip>


bool ANN::NeuralNetwork::Load(std::string filepath)
{
	std::ifstream file(filepath);
	if (!file.is_open()) return false;
	int buffer;
	const int CHAR_BUF_LEN = 100;
	char char_buffer[CHAR_BUF_LEN];
	file.getline(char_buffer, CHAR_BUF_LEN);
	std::string string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("activation type:"))
		throw "incorrect file format";
	file >> buffer;
	activation_type = (ActivationType)buffer;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("activation scale:"))
		throw "incorrect file format";
	file >> scale;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("configuration:"))
		throw "incorrect file format";
	file >> buffer;
	configuration.resize(buffer);
	for (size_t i = 0; i < configuration.size(); i++) {
		file >> configuration[i];
	}
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("weights:"))
		throw "incorrect file format";
	weights.resize(configuration.size() - 1);
	for (size_t i = 0; i < weights.size(); i++) {
		weights[i].resize(configuration[i + 1]);
		for (size_t j = 0; j < weights[i].size(); j++) {
			weights[i][j].resize(configuration[i]);
			for (size_t k = 0; k < weights[i][j].size(); k++) {
				file >> weights[i][j][k];
			}
		}
	}
	file.close();
	is_trained = true;
	return true;
}

bool ANN::NeuralNetwork::Save(std::string filepath)
{
	if (!is_trained) return false;
	std::ofstream file(filepath);
	if (!file.is_open()) return false;
	file << std::setprecision(9);
	file << "activation type:" << std::endl;
	file << (int)activation_type << std::endl;
	file << "activation scale:" << std::endl;
	file << scale << std::endl;
	file << "configuration:" << std::endl;
	file << configuration.size() << "\t";
	for each (int neuron_count in configuration) {
		file << neuron_count << "\t";
	}
	file << std::endl << "weights:" << std::endl;
	for each (auto weight_matrix in weights) {
		for each (auto weight_line in weight_matrix) {
			for each (auto weight in weight_line) {
				file << weight << " ";
			}
			file << std::endl;
		}
	}
	file.close();
	return true;
}

bool ANN::LoadData(
	std::string filepath,
	std::vector<std::vector<float>> & inputs,
	std::vector<std::vector<float>> & outputs)
{
	std::ifstream file(filepath);
	if (!file.is_open()) return false;
	int buffer;
	const int CHAR_BUF_LEN = 100;
	char char_buffer[CHAR_BUF_LEN];
	file.getline(char_buffer, CHAR_BUF_LEN);
	std::string string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("input_count:"))
		throw "incorrect file format";
	int input_count;
	file >> input_count;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("output_count:"))
		throw "incorrect file format";
	int output_count;
	file >> output_count;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("primer_count:"))
		throw "incorrect file format";
	int primer_count;
	file >> primer_count;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("data:"))
		throw "incorrect file format";
	inputs.resize(primer_count);
	outputs.resize(primer_count);
	//цикл по примерам
	for (int i = 0; i < primer_count; i++) {
		inputs[i].resize(input_count);
		//считываем входы
		for (int j = 0; j < input_count; j++) {
			file >> inputs[i][j];
		}
		file.getline(char_buffer, CHAR_BUF_LEN);
		//считываем выходы
		outputs[i].resize(output_count);
		for (int j = 0; j < output_count; j++) {
			file >> outputs[i][j];
		}
		file.getline(char_buffer, CHAR_BUF_LEN);
		file.getline(char_buffer, CHAR_BUF_LEN);
	}
	file.close();
	return true;
}

bool ANN::SaveData(
	std::string filepath,
	std::vector<std::vector<float>> & inputs,
	std::vector<std::vector<float>> & outputs)
{
	if (inputs.size() != outputs.size())
		throw "input size and output size must be the same";
	if (inputs.size() * outputs.size() == 0)
		throw "empty data";
	size_t input_count = inputs[0].size();
	size_t output_count = outputs[0].size();
	for (size_t i = 0; i < inputs.size(); i++) {
		if (inputs[i].size() != input_count)
			throw "incorrect input size";
		if (outputs[i].size() != output_count)
			throw "incorrect output size";
	}
	std::ofstream file(filepath);
	if (!file.is_open()) return false;
	file << std::setprecision(9);
	file << "input_count:" << std::endl;
	file << inputs[0].size() << std::endl;
	file << "output_count:" << std::endl;
	file << outputs[0].size() << std::endl;
	file << "primer_count:" << std::endl;
	file << inputs.size() << std::endl;
	file << "data:" << std::endl;
	for (int i = 0; i < inputs.size(); i++) {
		for (int j = 0; j < input_count; j++) {
			file << inputs[i][j] << "\t";
		}
		file << std::endl;
		for (int j = 0; j < output_count; j++) {
			file << outputs[i][j] << "\t";
		}
		file << std::endl;
		file << std::endl;
	}
	file.close();
	return true;
}

std::vector<int> ANN::NeuralNetwork::GetConfiguration()
{
	return configuration;
}

float ANN::NeuralNetwork::Activation(float neuronInput)
{
	if (activation_type == POSITIVE_SYGMOID) {
		return (1.f / (1.f + expf(-scale * neuronInput)));
	}
	else if (activation_type == BIPOLAR_SYGMOID) {
		return (2.f / (1.f + expf(-scale * neuronInput)) - 1.f);
	}
	return -1.f;
}

float ANN::NeuralNetwork::ActivationDerivative(float activation)
{
	if (activation_type == POSITIVE_SYGMOID) {
		return scale * activation * (1.f - activation);
	}
	else if (activation_type == BIPOLAR_SYGMOID) {
		return scale * 0.5f * (1.f + activation) * (1.f - activation);
	}
	return -1;
}

ANN::NeuralNetwork::~NeuralNetwork()
{}

std::string ANN::GetTestString()
{
	return "Hello! You succesfully plug ANN library!";
}
