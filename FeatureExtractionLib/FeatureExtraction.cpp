#define ANNDLL_EXPORTS
#include "FeatureExtraction.h"

std::string fe::GetTestString()
{
	return "You successfuly plug feature extraction library!";
}

std::shared_ptr<fe::IBlobProcessor> fe::CreateBlobProcessor()
{
	throw "Stub called!";
	return NULL;
}

std::shared_ptr<fe::PolynomialManager> fe::CreatePolynomialManager()
{
	throw "Stub called!";
	return NULL;
}