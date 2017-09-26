#define FEATURE_DLL_EXPORTS
#include "FeatureExtraction.h"
#include "PolynomialManagerImpl.h"
#include "BlobProcessorImpl.h"

FEATURE_DLL_API std::string fe::GetTestString()
{
	return "You successfuly plug feature extraction library!";
}

FEATURE_DLL_API std::shared_ptr<fe::IBlobProcessor> fe::CreateBlobProcessor()
{
    return std::make_shared<fe::BlobProcessorImpl>();
}

FEATURE_DLL_API std::shared_ptr<fe::PolynomialManager> fe::CreatePolynomialManager()
{
    return std::make_shared<fe::PolynomialManagerImpl>();
}