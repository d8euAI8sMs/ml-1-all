#pragma once

#include <vector>
#include <memory>

#include "ExportMacro.h"
#include "IBlobProcessor.h"
#include "PolynomialManager.h"
#include "ComplexMoments.h"

/*������� ������� ������� ������ �� ����������*/
namespace fe {
	/*�������� �������� ������*/
	FEATURE_DLL_API std::string GetTestString();
	/*������� ���������� ������� ��������*/
	FEATURE_DLL_API std::shared_ptr<IBlobProcessor> CreateBlobProcessor();
	/*������� ������, ������������� �� ������ � ����������*/
	FEATURE_DLL_API std::shared_ptr<PolynomialManager> CreatePolynomialManager();
};

