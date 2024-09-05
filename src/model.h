#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace ANN {
  class Model {
    private:
      const int& m; // reserved for num_hidden_

      int num_hidden_;              // m
      int num_parameter_;           // 4*m+1
      std::vector<double> weights_; // (VECTOR) size: 4*m+1 --> | bo_ | Ao_ | bi_ | Ai_ |

      bool x1_log_;
      bool x2_log_;
      bool y_log_;

      char* species_;

      double* Ao_; // (VECTOR) size: 1 * m
      double* bo_; // (SCALAR) size: 1 * 1
      double* Ai_; // (MATRIX) size: m * 2
      double* bi_; // (VECTOR) size: m * 1

      // f = Ao*xo + bo
      // xo = Trasfer(yi)
      // yi = Ai*x + bi

    public:
      Model();
      ~Model(){};

      bool Init(const std::string& filename);
      void Pred(const double* x, double* f) const;
      void Pred(const int n, const double* x, double* f) const;
      void Derivative(const double* x, double* dfdx) const;
      void Derivative(const double* x, double* f, double* dfdx) const;
      void Derivative2(const double* x, double* f, double* dfdx, double* dfdx2) const;

    private:
      inline double Transfer(const double input) const;
      inline double DiffTransfer(const double input) const;
      inline double Diff2Transfer(const double input) const;
  };

  const double R = 8.31446261815324; // Universal gas constant (J/K-mol)

  const double erg2J = 1.0E-04; // Convert erg/g to J/kg

  const double cm2erg = 1.9864468E-16; // Convert cm-1 to erg

  const double boltz = 1.380649E-23; // Boltzmann constant (J/K)

  extern std::string string_buffer;

  extern std::vector<std::shared_ptr<Model>> models;

  extern std::vector<std::string> mode_pack; // Energy modes (T, R, V, E)

  extern std::vector<std::string> species_pack; // Species names (37 species)

  extern std::vector<double> weight_pack; // Molecular weights of 37 species in (kg/mol)

  extern std::vector<std::vector<double>> thetv; // Vibrational characteristic temperature (K)

  extern std::vector<double> ev0; // Zero-vibrational energy (erg)

  extern std::vector<std::vector<double>> ge; // Electronic multiplicity of ground state

  extern std::vector<std::vector<double>> thetel; // Electronic characteristic temperature (K)
}
