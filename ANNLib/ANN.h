#pragma once

#ifdef ANNDLL_EXPORTS
#define ANNDLL_API __declspec(dllexport) 
#else
#define ANNDLL_API __declspec(dllimport) 
#endif

#include <vector>
#include <memory>

namespace ANN
{
	ANNDLL_API class NeuralNetwork
	{
	public:
		enum ActivationType
		{
			POSITIVE_SYGMOID,
			BIPOLAR_SYGMOID
		};
		/**��������� ��������� ���� �� �����. ���� ����������� ������� ������ Save
		 * @param filepath - ��� � ���� �� ����� � ������
		 * @return - ���������� ����������
		 */
		ANNDLL_API virtual bool Load(std::string filepath);
		/**��������� ��������� ���� � ����. ���� ����������� ������� ������ Load
		 * @param filepath - ��� � ���� �� ����� � ������
		 * @return - ���������� ����������
		 */
		ANNDLL_API virtual bool Save(std::string filepath);
		/**�������� ������������ ����.
		 * @return ������������ ���� - ������ - � ������ �������� �������� ���������� �������� � ����.
		 *			����� �������� ������������� ������ ����.
		 */
		ANNDLL_API virtual std::vector<int> GetConfiguration();

		/**************************************************************************/
		/**********************��� ��� ���� ����������� �����**********************/
		/**************************************************************************/
		/**�������� ������ � ����� ����
		 * @return �������� ����
		 */
		ANNDLL_API virtual std::string GetType() = 0;
		/**��������������� ����� �� ��������� �����
		 * @param input - ����, ����� ������ ��������������� ���������� �������� �� ������� ����
		 * @param output -�����, ����� ������ ��������������� ���������� �������� � �������� ����
		 */
		ANNDLL_API virtual std::vector<float> Predict(std::vector<float> & input) = 0;
		/**������� ����
		 * @param inputs - ����� ��� ��������
		 * @param outputs - ������ ��� ��������
		 * @param max_iters - ������������ ���������� �������� ��� ��������
		 * @param eps - ������� ������ �� ���� �������� ��� ������� ���������� ��������� ��������
		 * @param speed - �������� ��������
		 * @param std_dump - ���������� �� ���������� � �������� �������� � ����������� ����� ������?
		 */
		ANNDLL_API virtual float MakeTrain(
			std::vector<std::vector<float>> & inputs,
			std::vector<std::vector<float>> & outputs,
			int max_iters = 10000,
			float eps = 0.1,
			float speed = 0.1,
			bool std_dump = false
		) = 0;
		/***************************************************************************/
		/***************************************************************************/
		ANNDLL_API virtual ~NeuralNetwork();
	protected:
		/**���� ����*/
		std::vector<std::vector<std::vector<float> > > weights;
		/**������������ ����.
		 * ����� ������� � ������� ������������ ������ ����
		 * �������� - ���������� ��������
		 */
		std::vector<int> configuration;
		/**������� �� ����?*/
		bool is_trained;
		/**�������������� ����������� ��������� ��������*/
		float scale;
		/**��� ������������� �������*/
		ActivationType activation_type;
		/**��������� �������� ������������� �������
		 * @param neuron_input - ������� �������� �������
		 * @return - �������� ������������� ������
		 */
		ANNDLL_API float Activation(float neuron_input);
		/**��������� �������� ����������� ������������� �������
		 * @param activation - �������� ������������� ��������, ��� ������� ����� ��������� �����������
		 * @return - �������� ����������� ������������� ������
		 */
		ANNDLL_API float ActivationDerivative(float activation);
	};

	/**������� ��������� ����
	 * @param configuration - ������������ ��������� ����
	 */
	ANNDLL_API std::shared_ptr<ANN::NeuralNetwork> CreateNeuralNetwork(
		std::vector<int> & configuration = std::vector<int>(),
		NeuralNetwork::ActivationType activation_type = NeuralNetwork::POSITIVE_SYGMOID
	);

	/**�������� ������� ��� �������� ����������� ����������
	 * @return ������ � ��������������
	 */
	ANNDLL_API std::string GetTestString();

	/**������� ������ �� �����
	 * @param filepath - ���� � ��� � ����� � �������.
	 * @param inputs - ����� ��� ������ ������
	 * @param outputs - ����� ��� ������ �������
	 * @return - ���������� ������
	 */
	ANNDLL_API bool LoadData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);

	/**�������� ������ � ����
	* @param filepath - ���� � ��� � ����� � �������.
	* @param inputs - ����� ��� ������
	* @param outputs - ������ ��� ������
	* @return - ���������� ������
	*/
	ANNDLL_API bool SaveData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);
}