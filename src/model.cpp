#include "ann.h"
#include "model.h"

#define ANN_MODEL_ATOM(model, ispecies) \
  double ANN_##model##T(const double x1, const double x2) { \
    using namespace ANN; \
    return 1.5*R*x1/weight_pack[ispecies]; \
  } \
  double ANN_##model##T_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    grad[0] = 1.5*R/weight_pack[ispecies]; \
    grad[1] = 0.0; \
    return grad[0]*x1; \
  } \
  double ANN_##model##R(const double x1, const double x2) { \
    return 0.0; \
  } \
  double ANN_##model##R_Grad(double* grad, const double x1, const double x2) { \
    grad[0] = grad[1] = 0.0; \
    return 0.0; \
  } \
  double ANN_##model##V(const double x1, const double x2) { \
    return 0.0; \
  } \
  double ANN_##model##V_Grad(double* grad, const double x1, const double x2) { \
    grad[0] = grad[1] = 0.0; \
    return 0.0; \
  } \
  double ANN_##model##E(const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies]->Pred(x, &lnE); \
    return erg2J*exp(lnE); \
  } \
  double ANN_##model##E_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies]->Derivative(x, &lnE, grad); \
    const double y = erg2J*exp(lnE); \
    grad[0] *= y; \
    grad[1] *= y; \
    return y; \    
  }

#define ANN_MODEL_MOLECULE(model, ispecies) \
  double ANN_##model##T(const double x1, const double x2) { \
    using namespace ANN; \
    return 1.5*R*x1/weight_pack[ispecies]; \
  } \
  double ANN_##model##T_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    grad[0] = 1.5*R/weight_pack[ispecies]; \
    grad[1] = 0.0; \
    return grad[0]*x1; \
  } \
  double ANN_##model##R(const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[3*ispecies + 0 - 20]->Pred(x, &lnE); \
    return erg2J*exp(lnE); \
  } \
  double ANN_##model##R_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[3*ispecies + 0 - 20]->Derivative(x, &lnE, grad); \
    const double y = erg2J*exp(lnE); \
    grad[0] *= y; \
    grad[1] *= y; \
    return y; \
  } \
  double ANN_##model##V(const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[3*ispecies + 1 - 20]->Pred(x, &lnE); \
    return erg2J*exp(lnE); \
  } \
  double ANN_##model##V_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[3*ispecies + 1 - 20]->Derivative(x, &lnE, grad); \
    const double y = erg2J*exp(lnE); \
    grad[0] *= y; \
    grad[1] *= y; \
    return y; \
  } \
  double ANN_##model##E(const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[3*ispecies + 2 - 20]->Pred(x, &lnE); \
    return erg2J*exp(lnE); \
  } \
  double ANN_##model##E_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[3*ispecies + 2 - 20]->Derivative(x, &lnE, grad); \
    const double y = erg2J*exp(lnE); \
    grad[0] *= y; \
    grad[1] *= y; \
    return y; \
  }

#define ANN_MODEL_POLYATOM(model, ispecies) \
  double ANN_##model##T(const double x1, const double x2) { \
    using namespace ANN; \
    return 1.5*R*x1/weight_pack[ispecies]; \
  } \
  double ANN_##model##T_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    grad[0] = 1.5*R/weight_pack[ispecies]; \
    grad[1] = 0.0; \
    return grad[0]*x1; \
  } \
  double ANN_##model##R(const double x1, const double x2) { \
    using namespace ANN; \
    return R*x1/weight_pack[ispecies]; \
  } \
  double ANN_##model##R_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    grad[0] = R/weight_pack[ispecies]; \
    grad[1] = 0.0; \
    return grad[0]*x1; \
  } \
  double ANN_##model##V(const double x1, const double x2) { \
    using namespace ANN; \
    double E = 0.0; \
    const double Rs = R/weight_pack[ispecies]; \
    const double x2_inv = 1.0/x2; \
    for (int ivib = 0; ivib < thetv[ispecies].size(); ivib++) { \
      E += Rs*thetv[ispecies][ivib]/(exp(thetv[ispecies][ivib]*x2_inv) - 1.0); \
    } \
    return E; \
  } \
  double ANN_##model##V_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    double E = 0.0; \
    double Cv = 0.0; \
    const double Rs = R/weight_pack[ispecies]; \
    for (int ivib = 0; ivib < thetv[ispecies].size(); ivib++) { \
      const double Tratio = thetv[ispecies][ivib]/x2; \
      const double Tratio2 = Tratio*Tratio; \
      E += Rs*thetv[ispecies][ivib]/(exp(Tratio) - 1.0); \
      Cv += Rs*Tratio2*exp(Tratio)/((exp(Tratio) - 1.0)*(exp(Tratio) - 1.0)); \
    } \
    grad[0] = 0.0; \
    grad[1] = Cv; \
    return E; \
  } \
  double ANN_##model##E(const double x1, const double x2) { \
    using namespace ANN; \
    double qs = 0.0; \
    double qs1 = 0.0; \
    double qs2 = 0.0; \
    const double x2_inv = 1.0/x2; \
    for (int iele = 0; iele < thetel[ispecies].size(); iele++) { \
      qs = ge[ispecies][iele]*exp(-thetel[ispecies][iele]*x2_inv); \
      qs1 += qs*thetel[ispecies][iele]; \
      qs2 += qs; \
    } \
    const double E = qs1/qs2*R/weight_pack[ispecies]; \
    return E; \
  } \
  double ANN_##model##E_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double Rs = R/weight_pack[ispecies]; \
    const double x2_inv = 1.0/x2; \
    double qs = 0.0; \
    double qs1 = 0.0; \
    double qs2 = 0.0; \
    double qs3 = 0.0; \
    double qs4 = 0.0; \
    for (int iele = 0; iele < thetel[ispecies].size(); iele++) { \
      qs = ge[ispecies][iele]*exp(-thetel[ispecies][iele]*x2_inv); \
      qs1 += qs*thetel[ispecies][iele]; \
      qs2 += qs; \
      qs3 += qs*thetel[ispecies][iele]*thetel[ispecies][iele]*x2_inv*x2_inv; \
      qs4 += qs*thetel[ispecies][iele]*x2_inv*x2_inv; \
    } \
    const double cv = (qs3*qs2 - qs1*qs4)/qs2/qs2*Rs; \
    const double E = qs1/qs2*Rs; \
    grad[0] = 0.0; \
    grad[1] = cv; \
    return E; \
  }

#define ANN_MODEL_ELECTRON(model, ispecies) \
  double ANN_##model##T(const double x1, const double x2) { \
    using namespace ANN; \
    return 1.5*R*x2/weight_pack[ispecies]; \
  } \
  double ANN_##model##T_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    grad[0] = 0.0; \
    grad[1] = 1.5*R/weight_pack[ispecies]; \
    return grad[1]*x2; \
  } \
  double ANN_##model##R(const double x1, const double x2) { \
    return 0.0; \
  } \
  double ANN_##model##R_Grad(double* grad, const double x1, const double x2) { \
    grad[0] = grad[1] = 0.0; \
    return 0.0; \
  } \
  double ANN_##model##V(const double x1, const double x2) { \
    return 0.0; \
  } \
  double ANN_##model##V_Grad(double* grad, const double x1, const double x2) { \
    grad[0] = grad[1] = 0.0; \
    return 0.0; \
  } \
  double ANN_##model##E(const double x1, const double x2) { \
    return 0.0; \
  } \
  double ANN_##model##E_Grad(double* grad, const double x1, const double x2) { \
    grad[0] = grad[1] = 0.0; \
    return 0.0; \
  }

#define FPTR_RETURN(species_pack) \
  if (strcmp(species, #species_pack) == 0) { \
    if (strcmp(mode, "T") == 0) { \
      return &ANN_##species_pack##T; \
    } \
    else if (strcmp(mode, "R") == 0) { \
      return &ANN_##species_pack##R; \
    } \
    else if (strcmp(mode, "V") == 0) { \
      return &ANN_##species_pack##V; \
    } \
    else if (strcmp(mode, "E") == 0) { \
      return &ANN_##species_pack##E; \
    } \
  }

#define FPTR_GRAD_RETURN(species_pack) \
  if (strcmp(species, #species_pack) == 0) { \
    if (strcmp(mode, "T") == 0) { \
      return &ANN_##species_pack##T_Grad; \
    } \
    else if (strcmp(mode, "R") == 0) { \
      return &ANN_##species_pack##R_Grad; \
    } \
    else if (strcmp(mode, "V") == 0) { \
      return &ANN_##species_pack##V_Grad; \
    } \
    else if (strcmp(mode, "E") == 0) { \
      return &ANN_##species_pack##E_Grad; \
    } \
  }

ANN_MODEL_ATOM(N, 0);         // 1. N
ANN_MODEL_ATOM(O, 1);         // 2. O
ANN_MODEL_ATOM(C, 2);         // 3. C
ANN_MODEL_ATOM(H, 3);         // 4. H
ANN_MODEL_ATOM(Ar, 4);        // 5. Ar
ANN_MODEL_ATOM(Np, 5);        // 6. Np
ANN_MODEL_ATOM(Op, 6);        // 7. Op
ANN_MODEL_ATOM(Cp, 7);        // 8. Cp
ANN_MODEL_ATOM(Hp, 8);        // 9. Hp
ANN_MODEL_ATOM(Arp, 9);       // 10. Arp
ANN_MODEL_MOLECULE(N2, 10);   // 11. N2
ANN_MODEL_MOLECULE(O2, 11);   // 12. O2
ANN_MODEL_MOLECULE(C2, 12);   // 13. C2
ANN_MODEL_MOLECULE(H2, 13);   // 14. H2
ANN_MODEL_MOLECULE(NO, 14);   // 15. NO
ANN_MODEL_MOLECULE(NH, 15);   // 16. NH
ANN_MODEL_MOLECULE(OH, 16);   // 17. OH
ANN_MODEL_MOLECULE(CN, 17);   // 18. CN
ANN_MODEL_MOLECULE(CO, 18);   // 19. CO
ANN_MODEL_MOLECULE(CH, 19);   // 20. CH
ANN_MODEL_MOLECULE(SiO, 20);  // 21. SiO
ANN_MODEL_MOLECULE(N2p, 21);  // 22. N2p
ANN_MODEL_MOLECULE(O2p, 22);  // 23. O2p
ANN_MODEL_MOLECULE(NOp, 23);  // 24. NOp
ANN_MODEL_MOLECULE(CNp, 24);  // 25. CNp
ANN_MODEL_MOLECULE(COp, 25);  // 26. COp
ANN_MODEL_POLYATOM(C3, 26);   // 27. C3
ANN_MODEL_POLYATOM(CO2, 27);  // 28. CO2
ANN_MODEL_POLYATOM(C2H, 28);  // 29. C2H
ANN_MODEL_POLYATOM(CH2, 29);  // 30. CH2
ANN_MODEL_POLYATOM(H2O, 30);  // 31. H2O
ANN_MODEL_POLYATOM(HCN, 31);  // 32. HCN
ANN_MODEL_POLYATOM(CH3, 32);  // 33. CH3
ANN_MODEL_POLYATOM(CH4, 33);  // 34. CH4
ANN_MODEL_POLYATOM(C2H2, 34); // 35. C2H2
ANN_MODEL_POLYATOM(H2O2, 35); // 36. H2O2
ANN_MODEL_ELECTRON(e, 36);    // 37. e

ANN_DEF ANN_MODEL(const char* species, const char* mode) {
  FPTR_RETURN(N);    // 1. N
  FPTR_RETURN(O);    // 2. O
  FPTR_RETURN(C);    // 3. C
  FPTR_RETURN(H);    // 4. H
  FPTR_RETURN(Ar);   // 5. Ar
  FPTR_RETURN(Np);   // 6. Np
  FPTR_RETURN(Op);   // 7. Op
  FPTR_RETURN(Cp);   // 8. Cp
  FPTR_RETURN(Hp);   // 9. Hp
  FPTR_RETURN(Arp);  // 10. Arp
  FPTR_RETURN(N2);   // 11. N2
  FPTR_RETURN(O2);   // 12. O2
  FPTR_RETURN(C2);   // 13. C2
  FPTR_RETURN(H2);   // 14. H2
  FPTR_RETURN(NO);   // 15. NO
  FPTR_RETURN(NH);   // 16. NH
  FPTR_RETURN(OH);   // 17. OH
  FPTR_RETURN(CN);   // 18. CN
  FPTR_RETURN(CO);   // 19. CO
  FPTR_RETURN(CH);   // 20. CH
  FPTR_RETURN(SiO);  // 21. SiO
  FPTR_RETURN(N2p);  // 22. N2p
  FPTR_RETURN(O2p);  // 23. O2p
  FPTR_RETURN(NOp);  // 24. NOp
  FPTR_RETURN(CNp);  // 25. CNp
  FPTR_RETURN(COp);  // 26. COp
  FPTR_RETURN(C3);   // 27. C3
  FPTR_RETURN(CO2);  // 28. CO2
  FPTR_RETURN(C2H);  // 29. C2H
  FPTR_RETURN(CH2);  // 30. CH2
  FPTR_RETURN(H2O);  // 31. H2O
  FPTR_RETURN(HCN);  // 32. HCN
  FPTR_RETURN(CH3);  // 33. CH3
  FPTR_RETURN(CH4);  // 34. CH4
  FPTR_RETURN(C2H2); // 35. C2H2
  FPTR_RETURN(H2O2); // 36. H2O2
  FPTR_RETURN(e);    // 37. e
  return nullptr;
}

ANN_GRAD_DEF ANN_MODEL_GRAD(const char* species, const char* mode) {
  FPTR_GRAD_RETURN(N);    // 1. N
  FPTR_GRAD_RETURN(O);    // 2. O
  FPTR_GRAD_RETURN(C);    // 3. C
  FPTR_GRAD_RETURN(H);    // 4. H
  FPTR_GRAD_RETURN(Ar);   // 5. Ar
  FPTR_GRAD_RETURN(Np);   // 6. Np
  FPTR_GRAD_RETURN(Op);   // 7. Op
  FPTR_GRAD_RETURN(Cp);   // 8. Cp
  FPTR_GRAD_RETURN(Hp);   // 9. Hp
  FPTR_GRAD_RETURN(Arp);  // 10. Arp
  FPTR_GRAD_RETURN(N2);   // 11. N2
  FPTR_GRAD_RETURN(O2);   // 12. O2
  FPTR_GRAD_RETURN(C2);   // 13. C2
  FPTR_GRAD_RETURN(H2);   // 14. H2
  FPTR_GRAD_RETURN(NO);   // 15. NO
  FPTR_GRAD_RETURN(NH);   // 16. NH
  FPTR_GRAD_RETURN(OH);   // 17. OH
  FPTR_GRAD_RETURN(CN);   // 18. CN
  FPTR_GRAD_RETURN(CO);   // 19. CO
  FPTR_GRAD_RETURN(CH);   // 20. CH
  FPTR_GRAD_RETURN(SiO);  // 21. SiO
  FPTR_GRAD_RETURN(N2p);  // 22. N2p
  FPTR_GRAD_RETURN(O2p);  // 23. O2p
  FPTR_GRAD_RETURN(NOp);  // 24. NOp
  FPTR_GRAD_RETURN(CNp);  // 25. CNp
  FPTR_GRAD_RETURN(COp);  // 26. COp
  FPTR_GRAD_RETURN(C3);   // 27. C3
  FPTR_GRAD_RETURN(CO2);  // 28. CO2
  FPTR_GRAD_RETURN(C2H);  // 29. C2H
  FPTR_GRAD_RETURN(CH2);  // 30. CH2
  FPTR_GRAD_RETURN(H2O);  // 31. H2O
  FPTR_GRAD_RETURN(HCN);  // 32. HCN
  FPTR_GRAD_RETURN(CH3);  // 33. CH3
  FPTR_GRAD_RETURN(CH4);  // 34. CH4
  FPTR_GRAD_RETURN(C2H2); // 35. C2H2
  FPTR_GRAD_RETURN(H2O2); // 36. H2O2
  FPTR_GRAD_RETURN(e);    // 37. e
  return nullptr;
}

double ComputeEnergy(const ANN_DEF fptr, const double x1, const double x2) {
  return fptr(x1, x2);
}

double ComputeCv(const ANN_GRAD_DEF fptr, double* grad, const double x1, const double x2) {
  return fptr(&grad[0], x1, x2);
}

const char* ANN_Init(const char* modeldir) {
  using namespace ANN;

  const std::string header = std::string(modeldir);
  string_buffer = "ANN ver." + std::to_string(VER_MAJOR) + "." 
                             + std::to_string(VER_MINOR) + "." 
                             + std::to_string(VER_SUBMINOR) + "\n";

  models.resize(58);

  // Initialize models for atoms.
  int flag_cnt = 0;
  for (int ispecies = 0; ispecies < 10; ispecies++) {
    string_buffer += species_pack[ispecies] + "T";
    string_buffer += " (INIT SUCCESS)\n";
    models[ispecies] = std::make_shared<Model>();
    const bool init_flag = models[ispecies]->Init(header + "/" + species_pack[ispecies] + "/" + species_pack[ispecies] + mode_pack[2] + ".dat");
    string_buffer += species_pack[ispecies] + mode_pack[2];
    if (init_flag)
      string_buffer += " (INIT SUCCESS)\n";
    else
      string_buffer += " (INIT FAILURE)\n";
  }

  // Initialize models for diatomic molecules.
  for (int ispecies = 10; ispecies < 26; ispecies++) {
    string_buffer += species_pack[ispecies] + "T";
    string_buffer += " (INIT SUCCESS)\n";
    for (int imode = 0; imode < 3; imode++) {
      models[3*ispecies + imode - 20] = std::make_shared<Model>();
      const bool init_flag = models[3*ispecies + imode - 20]->Init(header + "/" + species_pack[ispecies] + "/" + species_pack[ispecies] + mode_pack[imode] + ".dat");
      string_buffer += species_pack[ispecies] + mode_pack[imode];
      if (init_flag)
        string_buffer += " (INIT SUCCESS)\n";
      else
        string_buffer += " (INIT FAILURE)\n";
    }
  }

  // Initialize models for polyatomic molecules.
  for (int ispecies = 26; ispecies < 36; ispecies++) {
    string_buffer += species_pack[ispecies] + "T";
    string_buffer += " (INIT SUCCESS)\n";
    for (int imode = 0; imode < 3; imode++) {
      string_buffer += species_pack[ispecies] + mode_pack[imode];
      string_buffer += " (INIT SUCCESS)\n";
    }
  }

  // Initialize models for electron.
  string_buffer += species_pack[36] + "T";
  for (int imode = 0; imode < 3; imode++) {
    string_buffer += species_pack[36] + mode_pack[imode];
    string_buffer += " (INIT SUCCESS)\n";
  }

  return string_buffer.c_str();
}

const char* ANN_Finalize(void) {
  using namespace ANN;

  string_buffer = "ANN model finalize!\n";

  for (int imodel = 0; imodel < 58; imodel++) {
    models[imodel].reset();
  }

  return string_buffer.c_str();
}

const char* ANN_Units(void) {
  using namespace ANN;

  string_buffer = 
    "Units used by ANN\n"
    "Mass: kg\n"
    "Temperature: K\n"
    "Energy per unit mass: J/kg\n";
    "\n";

  return string_buffer.c_str();
}

namespace ANN {
  std::string string_buffer;
  std::vector<std::shared_ptr<Model>> models;
  std::vector<std::string> mode_pack = {"R", "V", "E"};
  std::vector<std::string> species_pack = {"N"  , "O"  , "C"  , "H"  , "Ar"  , "Np"  , "Op", "Cp" , "Hp" , "Arp",
                                           "N2" , "O2" , "C2" , "H2" , "NO"  , "NH"  , "OH", "CN" , "CO" , "CH",
                                           "SiO", "N2p", "O2p", "NOp", "CNp" , "COp" , "C3", "CO2", "C2H", "CH2",
                                           "H2O", "HCN", "CH3", "CH4", "C2H2", "H2O2", "e"};
  std::vector<double> weight_pack = {
    1.400670E-02, // N
    1.599900E-02, // O
    1.201100E-02, // C
    1.008000E-03, // H
    3.994800E-02, // Ar
    1.400670E-02, // Np
    1.599940E-02, // Op
    1.201100E-02, // Cp
    1.008000E-03, // Hp
    3.994800E-02, // Arp
    2.801340E-02, // N2
    3.199800E-02, // O2
    2.402200E-02, // C2
    2.015880E-03, // H2
    3.000610E-02, // NO
    1.501468E-02, // NH
    1.700740E-02, // OH
    2.601744E-02, // CN
    2.801010E-02, // CO
    1.302000E-02, // CH
    4.408400E-02, // SiO
    2.801000E-02, // N2p
    3.199880E-02, // O2p
    3.000610E-02, // NOp
    2.616890E-02, // CNp
    2.800960E-02, // COp
    3.603300E-02, // C3
    4.400900E-02, // CO2
    2.502930E-02, // C2H
    1.402658E-02, // CH2
    1.801528E-02, // H2O
    2.702538E-02, // HCN
    1.503452E-02, // CH3
    1.604246E-02, // CH4
    2.604000E-02, // C2H2
    3.401400E-02, // H2O2
    5.485790E-07  // e 
  };

  std::vector<std::vector<double>> thetv = {
    {0.000000E+00},                                                                                                                 // N
    {0.000000E+00},                                                                                                                 // O
    {0.000000E+00},                                                                                                                 // C
    {0.000000E+00},                                                                                                                 // H
    {0.000000E+00},                                                                                                                 // Ar
    {0.000000E+00},                                                                                                                 // Np
    {0.000000E+00},                                                                                                                 // Op
    {0.000000E+00},                                                                                                                 // Cp
    {0.000000E+00},                                                                                                                 // Hp
    {0.000000E+00},                                                                                                                 // Arp
    {3.393500E+03},                                                                                                                 // N2
    {2.273576E+03},                                                                                                                 // O2
    {2.668550E+03},                                                                                                                 // C2
    {6.332449E+03},                                                                                                                 // H2
    {2.739662E+03},                                                                                                                 // NO
    {4.722518E+03},                                                                                                                 // NH
    {5.377918E+03},                                                                                                                 // OH
    {2.976280E+03},                                                                                                                 // CN
    {3.121919E+03},                                                                                                                 // CO
    {4.116036E+03},                                                                                                                 // CH
    {1.786348E+03},                                                                                                                 // SiO
    {3.175740E+03},                                                                                                                 // N2p
    {2.740576E+03},                                                                                                                 // O2p
    {3.419184E+03},                                                                                                                 // NOp
    {2.925145E+03},                                                                                                                 // CNp
    {3.185783E+03},                                                                                                                 // COp
    {3.058622E+02, 3.058622E+02, 2.479591E+03, 4.302663E+03},                                                                       // C3
    {1.341115E+03, 1.341115E+03, 2.753973E+03, 4.854217E+03},                                                                       // CO2
    {5.838549E+02, 5.944933E+02, 4.178792E+03, 6.938147E+03},                                                                       // C2H
    {2.089006E+03, 6.250888E+03, 6.740219E+03},                                                                                     // CH2
    {3.213415E+03, 7.657207E+03, 7.868867E+03},                                                                                     // H2O
    {1.537370E+03, 1.537370E+03, 4.404282E+03, 6.925168E+03},                                                                       // HCN
    {1.078861E+03, 2.813475E+03, 2.813475E+03, 6.221624E+03, 6.582388E+03, 6.582388E+03},                                           // CH3
    {2.688448E+03, 2.688448E+03, 2.688448E+03, 3.126155E+03, 3.126155E+03, 6.065759E+03, 6.277572E+03, 6.277572E+03, 6.277572E+03}, // CH4
    {1.298803E+03, 1.298803E+03, 1.550028E+03, 1.550028E+03, 4.136480E+03, 6.859573E+03, 7.065823E+03},                             // C2H2
    {4.050701E+02, 6.583442E+02, 9.295760E+02, 1.580336E+03, 1.945645E+03, 2.467823E+03, 3.641613E+03, 6.467357E+03, 6.478553E+03}, // H2O2
    {0.000000E+00}                                                                                                                  // e
  };

  std::vector<double> ev0 = { 
    0.00000000000000E+00, // N
    0.00000000000000E+00, // O
    0.00000000000000E+00, // C
    0.00000000000000E+00, // H
    0.00000000000000E+00, // Ar
    0.00000000000000E+00, // Np
    0.00000000000000E+00, // Op
    0.00000000000000E+00, // Cp
    0.00000000000000E+00, // Hp
    0.00000000000000E+00, // Arp
    5.02096554915455E+09, // N2
    2.94367586338399E+09, // O2
    4.60067167985591E+09, // C2
    1.29318612390291E+11, // H2
    3.78201033842404E+09, // NO
    1.29351873878448E+10, // NH
    1.29961141601302E+10, // OH
    4.74097831326855E+09, // CN
    4.62008788402121E+09, // CO
    1.30119190346340E+10, // CH
    1.68050134336794E+09, // SiO
    4.69552483397290E+09, // N2p
    3.54618906087543E+09, // O2p
    4.72113314831563E+09, // NOp
    4.65367328681488E+09, // CNp
    4.71214387206191E+09, // COp
    5.92902349970130E+09, // C3
    6.75611420186915E+09, // CO2
    1.41936936735291E+10, // C2H
    3.10640601979744E+10, // CH2
    3.00553759596447E+10, // H2O
    1.54000607577410E+10, // HCN
    5.01448979558475E+10, // CH3
    7.06316214008811E+10, // CH4
    2.63634627191810E+10, // C2H2
    2.08751310632062E+10, // H2O2
    0.00000000000000E+00  // e
  };

  std::vector<std::vector<double>> ge = {
    {1.0}, // N
    {1.0}, // O
    {1.0}, // C
    {1.0}, // H
    {1.0}, // Ar
    {1.0}, // Np
    {1.0}, // Op
    {1.0}, // Cp
    {1.0}, // Hp
    {1.0}, // Arp
    {1.0}, // N2
    {1.0}, // O2
    {1.0}, // C2
    {1.0}, // H2
    {1.0}, // NO
    {1.0}, // NH
    {1.0}, // OH
    {1.0}, // CN
    {1.0}, // CO
    {1.0}, // CH
    {1.0}, // SiO
    {1.0}, // N2p
    {1.0}, // O2p
    {1.0}, // NOp
    {1.0}, // CNp
    {1.0}, // COp
    {1.0, 6.0, 6.0, 3.0, 2.0, 6.0, 3.0, 1.0, 2.0, 2.0}, // C3
    {1.0, 3.0, 6.0, 3.0, 2.0}, // CO2
    {2.0, 4.0}, // C2H
    {3.0, 1.0}, // CH2
    {1.0}, // H2O
    {1.0, 1.0}, // HCN
    {2.0, 2.0}, // CH3
    {1.0, 1.0}, // CH4
    {2.0, 3.0, 6.0, 1.0, 3.0, 1.0}, // C2H2
    {1.0}, // H2O2
    {1.0}  // e
  };

  std::vector<std::vector<double>> thetel = {
    {0.0000000000000E+00}, // N
    {0.0000000000000E+00}, // O
    {0.0000000000000E+00}, // C
    {0.0000000000000E+00}, // H
    {0.0000000000000E+00}, // Ar
    {0.0000000000000E+00}, // Np
    {0.0000000000000E+00}, // Op
    {0.0000000000000E+00}, // Cp
    {0.0000000000000E+00}, // Hp
    {0.0000000000000E+00}, // Arp
    {0.0000000000000E+00}, // N2
    {0.0000000000000E+00}, // O2
    {0.0000000000000E+00}, // C2
    {0.0000000000000E+00}, // H2
    {0.0000000000000E+00}, // NO
    {0.0000000000000E+00}, // NH
    {0.0000000000000E+00}, // OH
    {0.0000000000000E+00}, // CN
    {0.0000000000000E+00}, // CO
    {0.0000000000000E+00}, // CH
    {0.0000000000000E+00}, // SiO
    {0.0000000000000E+00}, // N2p
    {0.0000000000000E+00}, // O2p
    {0.0000000000000E+00}, // NOp
    {0.0000000000000E+00}, // CNp
    {0.0000000000000E+00}, // COp
    {0.0000000000000E+00, 2.0142863611443E+04, 3.0933683403288E+04, 3.4242868139454E+04, 3.5502516503155E+04, 4.1868380792357E+04, 4.7191851889667E+04, 4.7335729486892E+04, 4.8486750264689E+04, 5.8270426875961E+04}, // C3
    {0.0000000000000E+00, 4.3163279167378E+04, 4.7479607084116E+04, 5.1795935000854E+04, 6.4744918751068E+04}, // CO2
    {0.0000000000000E+00, 5.7551038889838E+03}, // C2H
    {0.0000000000000E+00, 4.5278279846580E+03}, // CH2
    {0.0000000000000E+00}, // H2O
    {0.0000000000000E+00, 7.5185252716074E+04}, // HCN
    {0.0000000000000E+00, 6.6478643797624E+04}, // CH3
    {0.0000000000000E+00, 9.8887072572465E+04}, // CH4
    {0.0000000000000E+00, 3.5969399306149E+04, 5.0357159028608E+04, 6.0713468476835E+04, 7.1938798612298E+04, 7.7860800514062E+04}, // C2H2
    {0.0000000000000E+00}, // H2O2
    {0.0000000000000E+00}  // e
  };
}
