#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ann.h"
#include "model.h"

#define ANN_MODEL_INIT(model) \
  std::string model##_errlev = DEFAULT_ERROR_LEVEL; \
  if (accuracy_info.find("ALL") != accuracy_info.end()) \
    model##_errlev = accuracy_info.find("ALL")->second; \
  if (accuracy_info.find(#model) != accuracy_info.end()) \
    model##_errlev = accuracy_info.find(#model)->second; \
  model = std::make_shared<Model>(); \
  const bool model##_init = \
    model->Init(header + #model "_" + model##_errlev + "p.dat"); \
  string_buffer += (#model " " + model##_errlev + "% accuracy"); \
  if (model##_init) \
    string_buffer += " (INIT_SUCCESS) -> " + \
                     std::to_string(model->getAccuarcy()) + "% accuracy\n"; \
  else \
    string_buffer += " (INIT FAILURE)\n"

namespace ANN_internal { 
  std::vector<std::string> splitString(const std::string& str, const char delimiter) {
    std::vector<std::string> toks;
    std::stringstream ss(str);
    std::string tok;
    while (std::getline(ss, tok, delimiter)) toks.push_back(tok);
    if (str.back() == delimiter) toks.push_back("");
    return toks;
  }

  void removeString(std::string& str, const std::vector<char>& drops) {
    for (auto&& it = drops.begin(); it < drops.end(); it++)
      str.erase(std::remove(str.begin(), str.end(), *it), str.end());
  }
}

const char* ANN_Init(const char* modeldir, const char* errlev) {
  using namespace ANN;
  using namespace ANN_internal;

  std::unordered_map<std::string, std::string> accuracy_info;
  {
    std::string temp = std::string(errlev);
    removeString(temp, std::vector<char>{' ', '\0', '\n', '\r', '\t'});
    std::vector<std::string> values = splitString(temp, ',');
    for (auto&& v: values) {
      std::vector<std::string> pair = splitString(v, '=');
      if (pair.size() == 2) {
        accuracy_info[pair[0]] = pair[1];
      }
    }
  }

  const std::string header = std::string(modeldir);
  string_buffer = "ANN ver." + std::to_string(VER_MAJOR) + "." + std::to_string(VER_MINOR) + "." + std::to_string(VER_SUBMINOR) + "\n";

  ANN_MODEL_INIT(ET);
  ANN_MODEL_INIT(ER);
  ANN_MODEL_INIT(EV);
  ANN_MODEL_INIT(EE);

  return string_buffer.c_str();
}

const char* ANN_Finalize(void) {
  using namespace ANN;

  string_buffer = "ANN model finalize!\n";

  ET.reset();
  ER.reset();
  EV.reset();
  EE.reset();

  return string_buffer.c_str();
}

const char* ANN_Units(void) {
  using namespace ANN;

  string_buffer = 
    "Units used by ANN\n"
    "\n";

  return string_buffer.c_str();
}

namespace ANN {
  std::string string_buffer;
}

#define ANN_MODEL_DEF(x1, x2, y) \
  namespace ANN { \
    std::shared_ptr<Model> x1##x2##y \
  } \
  double ANN_##x1##x2##y(const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {std::log(x1), std::log(x2)}; \
    double ln##y; \
    x1##x2##y->Pred(x, &ln##y); \
    return std::exp(ln##y); \
  } \
  void ANN_##x1##x2##y##_batch(const int n, const double* x1, const double* x2, double* y) { \
    using namespace ANN; \
    std::vector<double> x(n * 2); \
    for (int i = 0; i < n; i++) { \
      x[i * 2] = std::log(x1[i]); \
      x[i * 2 + 1] = std::log(x2[i]); \
    } \
    double* ln##y = y; \
    x1##x2##y->Pred(n, &x[0], ln##y); \
    for (int i = 0; i < n; i++) \
      y[i] = std::exp(ln##y[i]); \
  }

ANN_MODEL_DEF(ET, Ttr, Tve);
