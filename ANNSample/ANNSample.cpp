#include <iostream>
#include <string>
#include <ANN.h>
using namespace std;
using namespace ANN;

const string input_file = "../data.dat";
const string network_file = "../network.dat";

void die(string message) { cout << message << endl; system("pause"); exit(1); }

template < typename T >
std::ostream & operator << (std::ostream & out, std::vector < T > v)
{
    out << "{";
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if ((i + 1) != v.size())
        {
            out << ", ";
        }
    }
    out << "}";
    return out;
}

int main()
{
	cout << "hello ANN!" << endl;
	cout << GetTestString().c_str() << endl;

    auto network = ANN::CreateNeuralNetwork();
    if (!network->Load(network_file))
    {
        die("cannot read network from file");
    }

    cout << "network type information:" << endl;
    cout << network->GetType() << endl;

    vector < vector < float > > input_data, output_data;

    if (!ANN::LoadData(input_file, input_data, output_data))
    {
        die("cannot load data from input file");
    }

    for (int i = 0; i < input_data.size(); ++i)
    {
        auto o = network->Predict(input_data[i]);
        cout << endl;
        cout << "input:    " << input_data[i] << endl;
        cout << "expected: " << output_data[i] << endl;
        cout << "output:   " << o << endl;
    }

	system("pause");
	return 0;
}