#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>

#include "ann.h"

int main(void) {
  // Print the list of units used by ANN
  const char* unit_info = ANN_Units();
  std::cout << std::string(unit_info) << std::endl;

  // Initialize ANN
  const std::string dir = "model/";
  const int nspecies = 1;
  //const char* species_pack[] = {"N2", "O2", "NO", "N", "O", "NOp", "N2p", "O2p", "Np", "Op", "e"};
  const char* species_pack[] = {"N2"};
  const char* mode_pack[4] = {"T", "R", "V", "E"};
  std::vector<ANN_DEF> fptrs;
  std::vector<ANN_GRAD_DEF> gradptrs;
  fptrs.resize(4*nspecies);
  gradptrs.resize(4*nspecies);
  const char* initlog = ANN_Init(dir.c_str());
  std::cout << std::string(initlog) << std::endl;

  for (int i = 0; i < nspecies; i++) {
    for (int j = 0; j < 4; j++) {
      fptrs[i*4 + j] = ANN_MODEL(species_pack[i], mode_pack[j]);
      gradptrs[i*4 + j] = ANN_MODEL_GRAD(species_pack[i], mode_pack[j]);
    }
  }

  // Usage example of ANN subroutines

  const double Ttr = 10000.0;
  const double Tve = 10000.0;

  double ET = 0.0; // Translational energy of N2 [J/kg]
  double ER = 0.0; // Rotational energy of N2 [J/kg]
  double EV = 0.0; // Vibrational energy of N2 [J/kg]
  double EE = 0.0; // Electronic energy of N2 [J/kg]

  // grad[0] = derivative of EV with respect to Ttr (dEV/dTtr)
  // grad[1] = derivative of EV with respect to Tve (dEV/dTve)
  std::vector<double> CvT(2, 0.0);
  std::vector<double> CvR(2, 0.0);
  std::vector<double> CvV(2, 0.0);
  std::vector<double> CvE(2, 0.0);

  std::cout << std::setprecision(8) << std::scientific << std::uppercase;

  for (int ispecies = 0; ispecies < nspecies; ispecies++)
  {
    std::ofstream tra_file("est_" + std::string(species_pack[ispecies]) + ".plt");
    std::ofstream rot_file("esr_" + std::string(species_pack[ispecies]) + ".plt");
    std::ofstream vib_file("esv_" + std::string(species_pack[ispecies]) + ".plt");
    std::ofstream ele_file("ese_" + std::string(species_pack[ispecies]) + ".plt");
    tra_file << "variables = \"Ttr\", \"Tve\", \"est\"\n";
    tra_file << "zone I=1000,J=1000\n";
    rot_file << "variables = \"Ttr\", \"Tve\", \"esr\"\n";
    rot_file << "zone I=1000,J=1000\n";
    vib_file << "variables = \"Ttr\", \"Tve\", \"esv\"\n";
    vib_file << "zone I=1000,J=1000\n";
    ele_file << "variables = \"Ttr\", \"Tve\", \"ese\"\n";
    ele_file << "zone I=1000,J=1000\n";

    for (int itemp1 = 0; itemp1 < 1000; itemp1++)
    {
      const double temp1 = 50.0 + double(itemp1)*50.0;
      for (int itemp2 = 0; itemp2 < 1000; itemp2++)
      {
        const double temp2 = 50.0 + double(itemp2)*50.0;
        ET = fptrs[ispecies*4 + 0](temp1, temp2);
        ER = fptrs[ispecies*4 + 1](temp1, temp2);
        EV = fptrs[ispecies*4 + 2](temp1, temp2);
        EE = fptrs[ispecies*4 + 3](temp1, temp2);
        tra_file << temp1 << "\t" << temp2 << "\t" << ET << "\n";
        rot_file << temp1 << "\t" << temp2 << "\t" << ER << "\n";
        vib_file << temp1 << "\t" << temp2 << "\t" << EV << "\n";
        ele_file << temp1 << "\t" << temp2 << "\t" << EE << "\n";
      }
    }
    tra_file.close();
    rot_file.close();
    vib_file.close();
    ele_file.close();
    //std::cout << "Translational-rotational temperature (Ttr) = " << Ttr << " K" << std::endl;
    //std::cout << "Vibrational-electronic temperature (Tve)   = " << Tve << " K" << std::endl;

    //// input: species, mode, Ttr, Tve
    //// output: energy
    //ET = fptrs[ispecies*4 + 0](Ttr, Tve);
    //ER = fptrs[ispecies*4 + 1](Ttr, Tve);
    //EV = fptrs[ispecies*4 + 2](Ttr, Tve);
    //EE = fptrs[ispecies*4 + 3](Ttr, Tve);
    //std::cout << std::endl << "Example 1, ANN_MODEL" << std::endl;
    //std::cout << "Translational energy (ET) = " << ET << " J/kg" << std::endl;
    //std::cout << "Rotational energy (ER)    = " << ER << " J/kg" << std::endl;
    //std::cout << "Vibrational energy (EV)   = " << EV << " J/kg" << std::endl;
    //std::cout << "Electronic energy (EE)    = " << EE << " J/kg" << std::endl;

    //// input: species, mode, Ttr, Tve
    //// output: specific heat
    //ET = gradptrs[ispecies*4 + 0](&CvT[0], Ttr, Tve);
    //ER = gradptrs[ispecies*4 + 1](&CvR[0], Ttr, Tve);
    //EV = gradptrs[ispecies*4 + 2](&CvV[0], Ttr, Tve);
    //EE = gradptrs[ispecies*4 + 3](&CvE[0], Ttr, Tve);
    //std::cout << std::endl << "Example 2, ANN_MODEL_Grad" << std::endl;
    //std::cout << "Translational energy (ET) = " << ET << " J/kg" << std::endl;
    //std::cout << "dET/dTtr                  = " << CvT[0] << " J/kg-K" << std::endl;
    //std::cout << "dET/dTve                  = " << CvT[1] << " J/kg-K" << std::endl;
    //std::cout << "Rotational energy (ER)    = " << ER << " J/kg" << std::endl;
    //std::cout << "dER/dTtr                  = " << CvR[0] << " J/kg-K" << std::endl;
    //std::cout << "dER/dTve                  = " << CvR[1] << " J/kg-K" << std::endl;
    //std::cout << "Vibrational energy (EV)   = " << EV << " J/kg" << std::endl;
    //std::cout << "dEV/dTtr                  = " << CvV[0] << " J/kg-K" << std::endl;
    //std::cout << "dEV/dTve                  = " << CvV[1] << " J/kg-K" << std::endl;
    //std::cout << "Electronic energy (EE)    = " << EE << " J/kg" << std::endl;
    //std::cout << "dEE/dTtr                  = " << CvE[0] << " J/kg-K" << std::endl;
    //std::cout << "dEE/dTve                  = " << CvE[1] << " J/kg-K" << std::endl;

    //std::cout << std::endl << std::endl;
  }

  // input: D, E
  // output: P, grad, hess
  //P = ANN_DEP_Hess(&hess[0], &grad[0], D, E);
  //std::cout << std::endl << "Example 3, ANN_DEP_Hess" << std::endl;
  //std::cout << "Pressure (P) = " << P << " Pa" << std::endl;
  //std::cout << "dP/dD = " << grad[0] << std::endl;
  //std::cout << "dP/dE = " << grad[1] << std::endl;
  //std::cout << "d**2P/dD**2 = " << hess[0] << std::endl;
  //std::cout << "d**2P/dDdE = " << hess[1] << std::endl;
  //std::cout << "d**2P/dE**2 = " << hess[2] << std::endl;

  // Finalize ANN
  const char* finlog = ANN_Finalize();
  std::cout << std::string(finlog) << std::endl;

  return 0;
}
