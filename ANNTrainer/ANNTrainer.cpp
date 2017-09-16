#include <iostream>
#include <string>
#include <ANN.h>
using namespace std;
using namespace ANN;

const string input_file = "../data.dat";
const string output_file = "../network.dat";

void die(string message) { cout << message << endl; system("pause"); exit(1); }

int main()
{
	cout << "hello ANN!" << endl;
	cout << GetTestString().c_str() << endl;

    vector < vector < float > > input_data, output_data;

    if (!ANN::LoadData(input_file, input_data, output_data))
    {
        die("cannot load data from input file");
    }

    vector < int > config({ 2, 10, 10, 1 });

    auto network = ANN::CreateNeuralNetwork(config);

    network->MakeTrain(input_data, output_data, 100000, 0.1, 0.1, true);

    cout << "network type information:" << endl;
    cout << network->GetType() << endl;

    if (!network->Save(output_file))
    {
        die("cannot save network to file");
    }

	system("pause");
	return 0;
}