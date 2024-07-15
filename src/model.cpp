#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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
    models[ispecies*3 + 2]->Pred(x, &lnE); \
    return erg2J*std::exp(lnE); \
  } \
  double ANN_##model##E_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies*3 + 2]->Derivative(x, &lnE, grad); \
    const double y = erg2J*std::exp(lnE); \
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
    models[ispecies*3 + 0]->Pred(x, &lnE); \
    return erg2J*std::exp(lnE); \
  } \
  double ANN_##model##R_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies*3 + 0]->Derivative(x, &lnE, grad); \
    const double y = erg2J*std::exp(lnE); \
    grad[0] *= y; \
    grad[1] *= y; \
    return y; \
  } \
  double ANN_##model##V(const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies*3 + 1]->Pred(x, &lnE); \
    return erg2J*std::exp(lnE); \
  } \
  double ANN_##model##V_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies*3 + 1]->Derivative(x, &lnE, grad); \
    const double y = erg2J*std::exp(lnE); \
    grad[0] *= y; \
    grad[1] *= y; \
    return y; \
  } \
  double ANN_##model##E(const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies*3 + 2]->Pred(x, &lnE); \
    return erg2J*std::exp(lnE); \
  } \
  double ANN_##model##E_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    const double x[2] = {x1, x2}; \
    double lnE; \
    models[ispecies*3 + 2]->Derivative(x, &lnE, grad); \
    const double y = erg2J*std::exp(lnE); \
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
    for (int ivib = 0; ivib < thetv[ispecies].size(); ivib++) { \
      E += R/weight_pack[ispecies]*thetv[ispecies][ivib]/(std::exp(thetv[ispecies][ivib]/x2) - 1.0); \
    } \
    return E; \
  } \
  double ANN_##model##V_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    double Cv = 0.0; \
    for (int ivib = 0; ivib < thetv[ispecies].size(); ivib++) { \
      const double arg1 = thetv[ispecies][ivib]/x2; \
      const double arg2 = arg1*arg1; \
      Cv += R/weight_pack[ispecies]*arg2*std::exp(arg1)/((std::exp(arg1) - 1.0)*(std::exp(arg1) - 1.0)); \
    } \
    return Cv; \
  } \
  double ANN_##model##E(const double x1, const double x2) { \
    using namespace ANN; \
    return 0.0; \
  } \
  double ANN_##model##E_Grad(double* grad, const double x1, const double x2) { \
    using namespace ANN; \
    grad[0] = grad[1] = 0.0; \
    return 0.0; \
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

  models.resize(3*36);
  for (int ispecies = 0; ispecies < 36; ispecies++) {
    string_buffer += species_pack[ispecies] + "T";
    string_buffer += " (INIT SUCCESS)\n";
    for (int imode = 0; imode < 3; imode++) {
      models[3*ispecies + imode] = std::make_shared<Model>();
      const bool init_flag = models[3*ispecies + imode]->Init(header + "/" + species_pack[ispecies] + "/" + species_pack[ispecies] + mode_pack[imode] + ".dat");
      string_buffer += species_pack[ispecies] + mode_pack[imode];
      if (init_flag)
        string_buffer += " (INIT SUCCESS)\n";
      else
        string_buffer += " (INIT FAILURE)\n";
    }
  }
  string_buffer += species_pack.back() + "T";
  string_buffer += " (INIT SUCCESS)\n";
  for (int imode = 0; imode < 3; imode++) {
    string_buffer += species_pack.back() + mode_pack[imode];
    string_buffer += " (INIT SUCCESS)\n";
  }

  return string_buffer.c_str();
}

const char* ANN_Finalize(void) {
  using namespace ANN;

  string_buffer = "ANN model finalize!\n";

  for (int ispecies = 0; ispecies < 36; ispecies++) {
    for (int imode = 0; imode < 3; imode++) {
      models[3*ispecies + imode].reset();
    }
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
  std::vector<double> weight_pack = {1.400670E-02, 1.599900E-02, 1.201100E-02, 1.008000E-03, 3.994800E-02, 1.400670E-02, 1.599940E-02, 1.201100E-02, 1.008000E-03, 3.994800E-02,
                                     2.801340E-02, 3.199800E-02, 2.402200E-02, 2.015880E-03, 3.000610E-02, 1.501468E-02, 1.700740E-02, 2.601744E-02, 2.801010E-02, 1.302000E-02,
                                     4.408400E-02, 2.801000E-02, 3.199880E-02, 3.000610E-02, 2.616890E-02, 2.800960E-02, 3.603300E-02, 4.400900E-02, 2.502930E-02, 1.402658E-02,
                                     1.801528E-02, 2.702538E-02, 1.503452E-02, 1.604246E-02, 2.604000E-02, 3.401400E-02, 5.485790E-07};

  std::vector<std::vector<double>> thetv = {{0.000000E+00}, {0.000000E+00}, {0.000000E+00}, {0.000000E+00}, {0.000000E+00}, {0.000000E+00}, {0.000000E+00}, {0.000000E+00}, {0.000000E+00}, {0.000000E+00},
                                            {3.393500E+03}, {2.273576E+03}, {2.668550E+03}, {6.332449E+03}, {2.739662E+03}, {4.722518E+03}, {5.377918E+03}, {2.976280E+03}, {3.121919E+03}, {4.116036E+03},
                                            {1.786348E+03}, {3.175740E+03}, {2.740576E+03}, {3.419184E+03}, {2.925145E+03}, {3.185783E+03},
                                            {3.058622E+02, 3.058622E+02, 2.479591E+03, 4.302663E+03}, {1.341115E+03, 1.341115E+03, 2.753973E+03, 4.854217E+03},
                                            {5.838549E+02, 5.944933E+02, 4.178792E+03, 6.938147E+03}, {2.089006E+03, 6.250888E+03, 6.740219E+03},
                                            {3.213415E+03, 7.657207E+03, 7.868867E+03}, {1.537370E+03, 1.537370E+03, 4.404282E+03, 6.925168E+03},
                                            {1.078861E+03, 2.813475E+03, 2.813475E+03, 6.221624E+03, 6.582388E+03, 6.582388E+03},
                                            {2.688448E+03, 2.688448E+03, 2.688448E+03, 3.126155E+03, 3.126155E+03, 6.065759E+03, 6.277572E+03, 6.277572E+03, 6.277572E+03},
                                            {1.298803E+03, 1.298803E+03, 1.550028E+03, 1.550028E+03, 4.136480E+03, 6.859573E+03, 7.065823E+03},
                                            {4.050701E+02, 6.583442E+02, 9.295760E+02, 1.580336E+03, 1.945645E+03, 2.467823E+03, 3.641613E+03, 6.467357E+03, 6.478553E+03}};
}
