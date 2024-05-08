#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ANN {
  class Model {
    private:
      const int& m; // reserved for num_hidden_

      int num_hidden_;
      int num_parameter_;
      std::vector<double> weights_;

      bool x1_log_;
      bool x2_log_;
      bool y_log_;
      double accuracy_;

      double* Ao_; // (VECTOR) 1 * m
      double* bo_; // (SCALAR) 1 * 1
      double* Ai_; // (MATRIX) m * 2
      double* bi_; // (VECTOR) m * 1

      // f = Ao*xo + bo
      // xo = Trasfer(yi)
      // yi = Ai*x + bi

    public:
      Model();
      ~Model();

      inline double GetAccuracy(void) const { return accuracy_; };

      bool Init(const std::string& filename);
      void Pred(const double* x, double* f) const;
      void Pred(const int n, const double* x, double* f) const;
      void Derivative(const double* x, double* dfdx) const;
      void Derivative(const double* x, double* f, double* dfdx) const;
      void Derivative2(const double* x, double* f, double* dfdx, double* dfdx2) const;

    private:
      inline double Transfer(const double input) const;
      inline double DiffTrasfer(const double input) const;
      inline double Diff2Trasfer(const double input) const;
  };

  extern std::string string_buffer;

  extern std::shared_ptr<Model> ET;
  extern std::shared_ptr<Model> ER;
  extern std::shared_ptr<Model> EV;
  extern std::shared_ptr<Model> EE;

}
