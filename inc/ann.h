#pragma once

#define VER_MAJOR 1
#define VER_MNOR 0
#define VER_SUBMINOR 0

#ifdef _WINDLL
#define ANN_API(type) __declspec(dllexport) type
#else
#define ANN_API(type) type
#endif

#ifdef __cplusplus
extern "C" {
#endif

  ANN_API(const char*) ANN_Init(const char* modeldir, const char* errlev);
  ANN_API(const char*) ANN_Finalize(void);
  ANN_API(const char*) ANN_Units(void);

  ANN_API(double) ANN_ET(const char* species, const double Ttr, const double Tve);
  ANN_API(double) ANN_ER(const char* species, const double Ttr, const double Tve);
  ANN_API(double) ANN_EV(const char* species, const double Ttr, const double Tve);
  ANN_API(double) ANN_EE(const char* species, const double Ttr, const double Tve);

  ANN_API(double) ANN_Cvtt(const char* species, const double Ttr, const double Tve);
  ANN_API(double) ANN_Cvtv(const char* species, const double Ttr, const double Tve);
  ANN_API(double) ANN_Cvvt(const char* species, const double Ttr, const double Tve);
  ANN_API(double) ANN_Cvvv(const char* species, const double Ttr, const double Tve);

#ifdef __cplusplus
}
#endif
