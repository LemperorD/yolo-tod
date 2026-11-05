#pragma once

#include <memory>
#include <string>

#include "todrt/factory.hpp"

namespace todrt {

std::unique_ptr<Detector> CreateOpenVinoDetector(const std::string& model_name,
                                                 const DetectorOptions& opts,
                                                 const ModelRecipe& recipe,
                                                 const std::string& builder_name,
                                                 const std::string& preproc_name,
                                                 const std::string& postproc_name);

}
