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
		/**Прочитать нейронную сеть из файла. Сеть сохраняется вызовом метода Save
		 * @param filepath - имя и путь до файла с сеткой
		 * @return - успешность считывания
		 */
		ANNDLL_API virtual bool Load(std::string filepath);
		/**Сохранить нейронную сеть в файл. Сеть загружается вызовом метода Load
		 * @param filepath - имя и путь до файла с сеткой
		 * @return - успешность сохранения
		 */
		ANNDLL_API virtual bool Save(std::string filepath);
		/**Получить конфигурацию сети.
		 * @return конфигурация сети - массив - в каждом элементе хранится количество нейронов в слое.
		 *			Номер элемента соответствует номеру слоя.
		 */
		ANNDLL_API virtual std::vector<int> GetConfiguration();

		/**************************************************************************/
		/**********************ЭТО ВАМ НАДО РЕАЛИЗОВАТЬ САМИМ**********************/
		/**************************************************************************/
		/**Получить строку с типом сети
		 * @return описание сети
		 */
		ANNDLL_API virtual std::string GetType() = 0;
		/**Спрогнозировать выход по заданному входу
		 * @param input - вход, длина должна соответствовать количеству нейронов во входном слое
		 * @param output -выход, длина должна соответствовать количеству нейронов в выходном слое
		 */
		ANNDLL_API virtual std::vector<float> Predict(std::vector<float> & input) = 0;
		/**Обучить сеть
		 * @param inputs - входы для обучения
		 * @param outputs - выходы для обучения
		 * @param max_iters - максимальное количество итераций при обучении
		 * @param eps - средняя ошибка по всем примерам при которой происходит остановка обучения
		 * @param speed - скорость обучения
		 * @param std_dump - сбрасывать ли информацию о процессе обучения в стандартный поток вывода?
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
		/**Веса сети*/
		std::vector<std::vector<std::vector<float> > > weights;
		/**Конфигурация сети.
		 * номер элемета в массиве соответсвует номеру слоя
		 * значение - количеству нейронов
		 */
		std::vector<int> configuration;
		/**Обучена ли сеть?*/
		bool is_trained;
		/**Масштабирующий коэффициент аргумента сигмоиды*/
		float scale;
		/**Тип активационной функции*/
		ActivationType activation_type;
		/**Вычислить значение активационной функции
		 * @param neuron_input - входное значение нейрона
		 * @return - значение активационной фунции
		 */
		ANNDLL_API float Activation(float neuron_input);
		/**Вычислить значение производной активационной функции
		 * @param activation - значение активационной фнункции, для которой хотим вычислить производную
		 * @return - значение производной активационной фунции
		 */
		ANNDLL_API float ActivationDerivative(float activation);
	};

	/**Создать нейронную сеть
	 * @param configuration - конфигурация нейронной сети
	 */
	ANNDLL_API std::shared_ptr<ANN::NeuralNetwork> CreateNeuralNetwork(
		std::vector<int> & configuration = std::vector<int>(),
		NeuralNetwork::ActivationType activation_type = NeuralNetwork::POSITIVE_SYGMOID
	);

	/**Тестовая функция для проверки подключения библиотеки
	 * @return строка с поздравлениями
	 */
	ANNDLL_API std::string GetTestString();

	/**Считать данные из файла
	 * @param filepath - путь и имя к файлу с данными.
	 * @param inputs - буфер для записи входов
	 * @param outputs - буфер для записи выходов
	 * @return - успешность чтения
	 */
	ANNDLL_API bool LoadData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);

	/**Записать данные в файл
	* @param filepath - путь и имя к файлу с данными.
	* @param inputs - входы для записи
	* @param outputs - выходы для записи
	* @return - успешность записи
	*/
	ANNDLL_API bool SaveData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);
}