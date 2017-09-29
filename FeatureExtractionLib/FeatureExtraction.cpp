#define FEATURE_DLL_EXPORTS
#include "FeatureExtraction.h"
#include "PolynomialManagerImpl.h"
#include "BlobProcessorImpl.h"

std::string fe::GetTestString()
{
	return "You successfuly plug feature extraction library!";
}

std::shared_ptr<fe::IBlobProcessor> fe::CreateBlobProcessor()
{
    return std::make_shared<fe::BlobProcessorImpl>();
}

std::shared_ptr<fe::PolynomialManager> fe::CreatePolynomialManager()
{
    return std::make_shared<fe::PolynomialManagerImpl>();
}