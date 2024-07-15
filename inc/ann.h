#pragma once

#define VER_MAJOR 1
#define VER_MINOR 0
#define VER_SUBMINOR 0

#ifdef _WINDLL
#define ANN_API(type) __declspec(dllexport) type
#else
#define ANN_API(type) type
#endif

#ifdef __cplusplus
extern "C" {
#endif

  typedef double (*ANN_DEF)(const double, const double);
  typedef double (*ANN_GRAD_DEF)(double*, const double, const double);

  ANN_API(const char*) ANN_Init(const char* modeldir);
  ANN_API(const char*) ANN_Finalize(void);
  ANN_API(const char*) ANN_Units(void);

  ANN_API(ANN_DEF) ANN_MODEL(const char* species, const char* mode);
  ANN_API(ANN_GRAD_DEF) ANN_MODEL_GRAD(const char* species, const char* mode);

  ANN_API(double) ComputeEnergy(const ANN_DEF fptr, const double x1, const double x2);
  ANN_API(double) ComputeCv(const ANN_GRAD_DEF fptr, double* grad, const double x1, const double x2);

#ifdef __cplusplus
}
#endif
