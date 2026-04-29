#include "SIREN/interactions/CharmMesonDecay.h"

#include <cmath>

#include <rk/rk.hh>
#include <rk/geom3.hh>

#include "SIREN/dataclasses/InteractionRecord.h"     // for Interac...
#include "SIREN/dataclasses/InteractionSignature.h"  // for Interac...
#include "SIREN/dataclasses/Particle.h"              // for Particle
#include "SIREN/math/Vector3D.h"                     // for Vector3D
#include "SIREN/utilities/Constants.h"               // for GeV, pi
#include "SIREN/utilities/Random.h"                  // for SIREN_random

#include "SIREN/utilities/Integration.h"          // for rombergInt...
#include "SIREN/utilities/Interpolator.h"

#include "SIREN/interactions/Decay.h"

namespace siren {
namespace interactions {

CharmMesonDecay::CharmMesonDecay() {
  // this is the default initialization but should never be used

  // we need to compute cdf here b/c otherwise SampleFinalState becomes non constant
  // in this case, we will need to compute all the possible dGamma's here
  // maybe add a map here, but for now we hard code first
  std::vector<double> constants;
  constants.resize(3);
  // check the primary and secondaries of the signature
  constants[0] = 0.725; // this is f^+(0)|V_cs| for charged D
  constants[1] = 0.44; // this is alpha, same for all K final states
  constants[2] = 2.01027; // this is excited charged D meson

  double mD = particleMass(siren::dataclasses::Particle::ParticleType::DPlus);
  double mK = particleMass(siren::dataclasses::Particle::ParticleType::K0Bar);

  computeDiffGammaCDF(constants, mD, mK);
}

CharmMesonDecay::CharmMesonDecay(siren::dataclasses::Particle::ParticleType primary) {

  //standard stuff, constant across primary types
  std::vector<double> constants;
  constants.resize(3);
  double mD;
  double mK;

  if (primary == siren::dataclasses::Particle::ParticleType::DPlus) {
    constants[0] = 0.725; // this is f^+(0)|V_cs| for charged D
    constants[1] = 0.44; // this is alpha, same for all K final states
    constants[2] = 2.01027; // this is excited charged D meson

    mD = particleMass(siren::dataclasses::Particle::ParticleType::DPlus);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::K0Bar);

  } else if (primary == siren::dataclasses::Particle::ParticleType::D0) {
    constants[0] = 0.719; // this is f^+(0)|V_cs| for charged D
    constants[1] = 0.50; // this is alpha, same for all K final states
    constants[2] = 2.00697; // this is excited charged D meson

    mD = particleMass(siren::dataclasses::Particle::ParticleType::D0);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::KMinus);
  } else if (primary == siren::dataclasses::Particle::ParticleType::DMinus) {
    // CP-conjugate of D+: same form-factor constants, kaon charge flipped.
    constants[0] = 0.725; // this is f^+(0)|V_cs| for charged D
    constants[1] = 0.44; // this is alpha, same for all K final states
    constants[2] = 2.01027; // this is excited charged D meson

    mD = particleMass(siren::dataclasses::Particle::ParticleType::DMinus);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::K0); // not K0Bar
  } else if (primary == siren::dataclasses::Particle::ParticleType::D0Bar) {
    // CP-conjugate of D0: same form-factor constants, kaon charge flipped.
    constants[0] = 0.719; // this is f^+(0)|V_cs| for charged D
    constants[1] = 0.50; // this is alpha, same for all K final states
    constants[2] = 2.00697; // this is excited charged D meson

    mD = particleMass(siren::dataclasses::Particle::ParticleType::D0Bar);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::KPlus); // not KMinus
  }

  computeDiffGammaCDF(constants, mD, mK);

}

CharmMesonDecay::CharmMesonDecay(siren::dataclasses::Particle::ParticleType primary, bool force_muonic)
    : force_muonic_(force_muonic) {

  std::vector<double> constants;
  constants.resize(3);
  double mD;
  double mK;

  if (primary == siren::dataclasses::Particle::ParticleType::DPlus) {
    constants[0] = 0.725;
    constants[1] = 0.44;
    constants[2] = 2.01027;
    mD = particleMass(siren::dataclasses::Particle::ParticleType::DPlus);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::K0Bar);
  } else if (primary == siren::dataclasses::Particle::ParticleType::D0) {
    constants[0] = 0.719;
    constants[1] = 0.44;
    constants[2] = 2.00697;
    mD = particleMass(siren::dataclasses::Particle::ParticleType::D0);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::KMinus);
  } else if (primary == siren::dataclasses::Particle::ParticleType::DMinus) {
    // CP-conjugate of D+: same form-factor constants, kaon charge flipped.
    constants[0] = 0.725;
    constants[1] = 0.44;
    constants[2] = 2.01027;
    mD = particleMass(siren::dataclasses::Particle::ParticleType::DMinus);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::K0); // not K0Bar
  } else if (primary == siren::dataclasses::Particle::ParticleType::D0Bar) {
    // CP-conjugate of D0: same form-factor constants, kaon charge flipped.
    constants[0] = 0.719;
    constants[1] = 0.44;
    constants[2] = 2.00697;
    mD = particleMass(siren::dataclasses::Particle::ParticleType::D0Bar);
    mK = particleMass(siren::dataclasses::Particle::ParticleType::KPlus); // not KMinus
  }

  computeDiffGammaCDF(constants, mD, mK);
}

bool CharmMesonDecay::equal(Decay const & other) const {
    const CharmMesonDecay* x = dynamic_cast<const CharmMesonDecay*>(&other);

    if(!x)
        return false;
    else
        return primary_types == x->primary_types && force_muonic_ == x->force_muonic_;
}


double CharmMesonDecay::particleMass(siren::dataclasses::ParticleType particle) {
    switch(particle){
			case siren::dataclasses::ParticleType::D0:
				return( siren::utilities::Constants::D0Mass);
			case siren::dataclasses::ParticleType::D0Bar:
				return( siren::utilities::Constants::D0Mass);
			case siren::dataclasses::ParticleType::DPlus:
				return( siren::utilities::Constants::DPlusMass);
			case siren::dataclasses::ParticleType::DMinus:
				return( siren::utilities::Constants::DPlusMass);
			case siren::dataclasses::ParticleType::K0:
				return( siren::utilities::Constants::K0Mass);
			case siren::dataclasses::ParticleType::K0Bar:
				return( siren::utilities::Constants::K0Mass);
			case siren::dataclasses::ParticleType::KPlus:
				return( siren::utilities::Constants::KplusMass);
			case siren::dataclasses::ParticleType::KMinus:
				return( siren::utilities::Constants::KminusMass);
      case siren::dataclasses::ParticleType::EPlus:
        return( siren::utilities::Constants::electronMass );
      case siren::dataclasses::ParticleType::EMinus:
        return( siren::utilities::Constants::electronMass );
      case siren::dataclasses::ParticleType::MuPlus:
        return( siren::utilities::Constants::muonMass );
      case siren::dataclasses::ParticleType::MuMinus:
        return( siren::utilities::Constants::muonMass );
      case siren::dataclasses::ParticleType::TauPlus:
        return( siren::utilities::Constants::tauMass );
      case siren::dataclasses::ParticleType::TauMinus:
        return( siren::utilities::Constants::tauMass );
      default:
        return(0.0);
    }
}

double CharmMesonDecay::TotalDecayWidth(dataclasses::InteractionRecord const & record) const {
    return TotalDecayWidth(record.signature.primary_type);
}

// Always use the full decay width (all channels) for correct physical decay rate,
// even when force_muonic_ is set. This ensures the D-meson decay vertex is sampled correctly.
double CharmMesonDecay::TotalDecayWidth(siren::dataclasses::Particle::ParticleType primary) const {
    double total_width = 0;
    std::vector<dataclasses::InteractionSignature> all_signatures = GetAllSignaturesFromParent(primary);
    for (auto sig : all_signatures) {
      siren::dataclasses::InteractionRecord fake_record;
      fake_record.signature = sig;
      double this_width = TotalDecayWidthForFinalState(fake_record);
      total_width += this_width;
    }
    return total_width;
}

// current problem: in implementation we see kaons and pions both as hadrons, but they should have different branching ratios and form factors
double CharmMesonDecay::TotalDecayWidthForFinalState(dataclasses::InteractionRecord const & record) const {
    double branching_ratio = 0;  // init closes latent UB on unhandled-primary fallthrough
    double tau = 0;              // same safety init
    // read in the signature and types
    siren::dataclasses::Particle::ParticleType primary = record.signature.primary_type;
    std::vector<siren::dataclasses::Particle::ParticleType> secondaries_vector = record.signature.secondary_types;
    std::set<siren::dataclasses::Particle::ParticleType> secondaries = std::set<siren::dataclasses::Particle::ParticleType>(secondaries_vector.begin(), secondaries_vector.end());

    // define the decay modes
    std::set<siren::dataclasses::Particle::ParticleType> k0_eplus_nue = {siren::dataclasses::Particle::ParticleType::K0Bar,
                                                                            siren::dataclasses::Particle::ParticleType::EPlus,
                                                                            siren::dataclasses::Particle::ParticleType::NuE};
    std::set<siren::dataclasses::Particle::ParticleType> kminus_eplus_nue = {siren::dataclasses::Particle::ParticleType::KMinus,
                                                                            siren::dataclasses::Particle::ParticleType::EPlus,
                                                                            siren::dataclasses::Particle::ParticleType::NuE};
    std::set<siren::dataclasses::Particle::ParticleType> k0_muplus_numu = {siren::dataclasses::Particle::ParticleType::K0Bar,
                                                                            siren::dataclasses::Particle::ParticleType::MuPlus,
                                                                            siren::dataclasses::Particle::ParticleType::NuMu};
    std::set<siren::dataclasses::Particle::ParticleType> kminus_muplus_numu = {siren::dataclasses::Particle::ParticleType::KMinus,
                                                                            siren::dataclasses::Particle::ParticleType::MuPlus,
                                                                            siren::dataclasses::Particle::ParticleType::NuMu};
    // CP-conjugate decay-mode sets for DMinus / D0Bar
    std::set<siren::dataclasses::Particle::ParticleType> k0_eminus_nuebar = {siren::dataclasses::Particle::ParticleType::K0,
                                                                            siren::dataclasses::Particle::ParticleType::EMinus,
                                                                            siren::dataclasses::Particle::ParticleType::NuEBar};
    std::set<siren::dataclasses::Particle::ParticleType> k0_muminus_numubar = {siren::dataclasses::Particle::ParticleType::K0,
                                                                            siren::dataclasses::Particle::ParticleType::MuMinus,
                                                                            siren::dataclasses::Particle::ParticleType::NuMuBar};
    std::set<siren::dataclasses::Particle::ParticleType> kplus_eminus_nuebar = {siren::dataclasses::Particle::ParticleType::KPlus,
                                                                            siren::dataclasses::Particle::ParticleType::EMinus,
                                                                            siren::dataclasses::Particle::ParticleType::NuEBar};
    std::set<siren::dataclasses::Particle::ParticleType> kplus_muminus_numubar = {siren::dataclasses::Particle::ParticleType::KPlus,
                                                                            siren::dataclasses::Particle::ParticleType::MuMinus,
                                                                            siren::dataclasses::Particle::ParticleType::NuMuBar};
    std::set<siren::dataclasses::Particle::ParticleType> hadrons = {siren::dataclasses::Particle::ParticleType::Hadrons};
    if (primary == siren::dataclasses::Particle::ParticleType::DPlus) {
      tau = 1040 * (1e-15);
      if (secondaries == k0_eplus_nue) {branching_ratio = .1607;} // e+ semileptonic mode according to pdg
      else if (secondaries == k0_muplus_numu) {branching_ratio = .176;} // mu+ anything according to pdg
      else if (secondaries == hadrons) {branching_ratio = (1 - .1607 - .176);} // everything else
    } else if (primary == siren::dataclasses::Particle::ParticleType::D0) {
      tau = 410.1 * (1e-15);
      if (secondaries == kminus_eplus_nue) {branching_ratio = .0649;} // e+ semileptonic mode according to pdg
      else if (secondaries == kminus_muplus_numu) {branching_ratio = .067;} // mu+ anything according to pdg
      else if (secondaries == hadrons) {branching_ratio = (1 - .0649 - .067);} // everything else
    } else if (primary == siren::dataclasses::Particle::ParticleType::DMinus) {
      // CP-conjugate of DPlus; same lifetime & BRs.
      tau = 1040 * (1e-15);
      if (secondaries == k0_eminus_nuebar) {branching_ratio = .1607;}
      else if (secondaries == k0_muminus_numubar) {branching_ratio = .176;}
      else if (secondaries == hadrons) {branching_ratio = (1 - .1607 - .176);}
    } else if (primary == siren::dataclasses::Particle::ParticleType::D0Bar) {
      // CP-conjugate of D0; same lifetime & BRs.
      tau = 410.1 * (1e-15);
      if (secondaries == kplus_eminus_nuebar) {branching_ratio = .0649;}
      else if (secondaries == kplus_muminus_numubar) {branching_ratio = .067;}
      else if (secondaries == hadrons) {branching_ratio = (1 - .0649 - .067);}
    }
    else {
        std::cout << "this decay mode is not yet implemented!" << std::endl;
    }
    return branching_ratio * siren::utilities::Constants::hbar / tau * siren::utilities::Constants::GeV;
}


std::vector<dataclasses::InteractionSignature> CharmMesonDecay::GetPossibleSignatures() const {
    std::vector<dataclasses::InteractionSignature> signatures;
    for(auto primary : primary_types) {
      std::vector<dataclasses::InteractionSignature> new_signatures = GetPossibleSignaturesFromParent(primary);
      signatures.insert(signatures.end(),new_signatures.begin(),new_signatures.end());
    }
    return signatures;
}

// Returns all decay channels regardless of force_muonic_ setting.
std::vector<dataclasses::InteractionSignature> CharmMesonDecay::GetAllSignaturesFromParent(siren::dataclasses::Particle::ParticleType primary) const {
    std::vector<dataclasses::InteractionSignature> signatures;
    dataclasses::InteractionSignature semilep_signature;
    semilep_signature.primary_type = primary;
    semilep_signature.target_type = siren::dataclasses::Particle::ParticleType::Decay;
    semilep_signature.secondary_types.resize(3);
    dataclasses::InteractionSignature hadron_signature;
    hadron_signature.primary_type = primary;
    hadron_signature.target_type = siren::dataclasses::Particle::ParticleType::Decay;
    hadron_signature.secondary_types.resize(1);

    if (primary==siren::dataclasses::Particle::ParticleType::DPlus) {
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::K0Bar;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::EPlus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuE;
      signatures.push_back(semilep_signature);
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::K0Bar;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuPlus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMu;
      signatures.push_back(semilep_signature);
      hadron_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::Hadrons;
      signatures.push_back(hadron_signature);
    } else if (primary==siren::dataclasses::Particle::ParticleType::D0) {
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::KMinus;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::EPlus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuE;
      signatures.push_back(semilep_signature);
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::KMinus;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuPlus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMu;
      signatures.push_back(semilep_signature);
      hadron_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::Hadrons;
      signatures.push_back(hadron_signature);
    } else if (primary==siren::dataclasses::Particle::ParticleType::DMinus) {
      // CP-conjugate of DPlus: K0, l-, nubar
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::K0;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::EMinus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuEBar;
      signatures.push_back(semilep_signature);
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::K0;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuMinus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMuBar;
      signatures.push_back(semilep_signature);
      hadron_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::Hadrons;
      signatures.push_back(hadron_signature);
    } else if (primary==siren::dataclasses::Particle::ParticleType::D0Bar) {
      // CP-conjugate of D0: K+, l-, nubar
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::KPlus;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::EMinus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuEBar;
      signatures.push_back(semilep_signature);
      semilep_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::KPlus;
      semilep_signature.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuMinus;
      semilep_signature.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMuBar;
      signatures.push_back(semilep_signature);
      hadron_signature.secondary_types[0] = siren::dataclasses::Particle::ParticleType::Hadrons;
      signatures.push_back(hadron_signature);
    }
    else {
      std::cout << "this D meson decay has not been implemented yet" << std::endl;
    }
    return signatures;
}

// When force_muonic_ is true, returns only the muonic semileptonic channel.
std::vector<dataclasses::InteractionSignature> CharmMesonDecay::GetPossibleSignaturesFromParent(siren::dataclasses::Particle::ParticleType primary) const {
    if (!force_muonic_) {
        return GetAllSignaturesFromParent(primary);
    }
    // Only return K + mu + nu_mu channel
    std::vector<dataclasses::InteractionSignature> signatures;
    dataclasses::InteractionSignature sig;
    sig.primary_type = primary;
    sig.target_type = siren::dataclasses::Particle::ParticleType::Decay;
    sig.secondary_types.resize(3);
    // Lepton charge / neutrino flavour track the D-meson charm quantum number.
    // DPlus (c-cbar-d) and D0 (c-ubar) decay c -> s l+ nu_l  (mu+ nu_mu).
    // DMinus (cbar-d) and D0Bar (cbar-u) decay cbar -> sbar l- nubar_l (mu- nubar_mu).
    if (primary==siren::dataclasses::Particle::ParticleType::DPlus) {
      sig.secondary_types[0] = siren::dataclasses::Particle::ParticleType::K0Bar;
      sig.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuPlus;
      sig.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMu;
    } else if (primary==siren::dataclasses::Particle::ParticleType::D0) {
      sig.secondary_types[0] = siren::dataclasses::Particle::ParticleType::KMinus;
      sig.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuPlus;
      sig.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMu;
    } else if (primary==siren::dataclasses::Particle::ParticleType::DMinus) {
      sig.secondary_types[0] = siren::dataclasses::Particle::ParticleType::K0;
      sig.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuMinus;
      sig.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMuBar;
    } else if (primary==siren::dataclasses::Particle::ParticleType::D0Bar) {
      sig.secondary_types[0] = siren::dataclasses::Particle::ParticleType::KPlus;
      sig.secondary_types[1] = siren::dataclasses::Particle::ParticleType::MuMinus;
      sig.secondary_types[2] = siren::dataclasses::Particle::ParticleType::NuMuBar;
    }
    signatures.push_back(sig);
    return signatures;
}

std::vector<double> CharmMesonDecay::FormFactorFromRecord(dataclasses::CrossSectionDistributionRecord const & record) const {
  dataclasses::InteractionSignature signature = record.signature;
  std::vector<double> constants;
  constants.resize(3);
  // check the primary and secondaries of the signature
  if (signature.primary_type == dataclasses::Particle::ParticleType::DPlus && signature.secondary_types[0] == siren::dataclasses::Particle::ParticleType::K0Bar) {
    constants[0] = 0.725; // this is f^+(0)|V_cs| for charged D
    constants[1] = 0.44; // this is alpha, same for all K final states
    constants[2] = 2.01027; // this is excited charged D meson
  } else if (signature.primary_type == dataclasses::Particle::ParticleType::D0 && signature.secondary_types[0] == siren::dataclasses::Particle::ParticleType::KMinus) {
    constants[0] = 0.719; // this is f^+(0)|V_cs| for neutral D
    constants[1] = 0.50; // this is alpha, same for all K final states
    constants[2] = 2.00697; // this is excited neutral D meson
  } else if (signature.primary_type == dataclasses::Particle::ParticleType::DMinus && signature.secondary_types[0] == siren::dataclasses::Particle::ParticleType::K0) {
    // CP-conjugate of DPlus
    constants[0] = 0.725;
    constants[1] = 0.44;
    constants[2] = 2.01027;
  } else if (signature.primary_type == dataclasses::Particle::ParticleType::D0Bar && signature.secondary_types[0] == siren::dataclasses::Particle::ParticleType::KPlus) {
    // CP-conjugate of D0
    constants[0] = 0.719;
    constants[1] = 0.50;
    constants[2] = 2.00697;
  }
  return constants;
}

double CharmMesonDecay::DifferentialDecayWidth(dataclasses::InteractionRecord const & record) const {
    // first let the fully hadronic state be handled separately
    dataclasses::InteractionSignature signature = record.signature;
    if (signature.secondary_types[0] == siren::dataclasses::Particle::ParticleType::Hadrons) {
      return TotalDecayWidthForFinalState(record);
    }
    // get the form factor constants
    std::vector<double> constants = FormFactorFromRecord(record);
    // calculate the q^2
    rk::P4 pD(geom3::Vector3(record.primary_momentum[1], record.primary_momentum[2], record.primary_momentum[3]), record.primary_mass);
    rk::P4 pKPi(geom3::Vector3(record.secondary_momenta[0][1], record.secondary_momenta[0][2], record.secondary_momenta[0][3]), record.secondary_masses[0]);
    double Q2 = (pD - pKPi).dot(pD - pKPi);
    // primary and secondary masses are also needed
    double mD = record.primary_mass;
    double mK = record.secondary_masses[0];
    return DifferentialDecayWidth(constants, Q2, mD, mK);
}

double CharmMesonDecay::DifferentialDecayWidth(std::vector<double> constants, double Q2, double mD, double mK) const {
    // get the numerical constants from the vector
    double F0CKM = constants[0];
    double alpha = constants[1];
    double ms = constants[2];
    double Q2tilde = Q2 / (ms * ms);
    // compute the 3-momentum as a function of Q2
    // double EK = 0.5 * (Q2 - pow(mD, 2) + pow(mK, 2)) / mD; // energy of Kaon
    double EK = 0.5 * (pow(mD, 2) + pow(mK, 2) - Q2) / mD; // energy of Kaon
    if (EK * EK < mK * mK) return 0.0;

    double PK = pow(pow(EK, 2) - pow(mK, 2), 0.5);
    // plug in the constants
    double dGamma = pow(siren::utilities::Constants::FermiConstant,2) / (24 * pow(siren::utilities::Constants::pi,3)) * pow(F0CKM,2) *
                    pow((1/((1-Q2tilde) * (1 - alpha * Q2tilde))),2) * pow(PK,3);
    return dGamma;
}

void CharmMesonDecay::computeDiffGammaCDF(std::vector<double> constants, double mD, double mK) {

  // returns a 1D interpolater table for dGamma cdf
  // define the pdf with only Q2 as the input
  std::function<double(double)> pdf = [&] (double x) -> double {
            return DifferentialDecayWidth(constants, x, mD, mK);
        };
  // first normalize the integral
  double min = 0;
  double max = 1.4; // these set the min and max of the Q2 considered
  double normalization = siren::utilities::rombergIntegrate(pdf, min, max);
  std::function<double(double)> normed_pdf = [&] (double x) -> double {
            return DifferentialDecayWidth(constants, x, mD, mK) / normalization;
        };
  // now create the spline and compute the CDF

  // set the Q2 nodes (use 100 nodes)
  std::vector<double> Q2spline;
  for (int i = 0; i < 100; ++i) {
      Q2spline.push_back(0.01 + i * (max-min) / 100 );
  }

  // declare the cdf vectors
  std::vector<double> cdf_vector;
  std::vector<double> cdf_Q2_nodes;
  std::vector<double> pdf_vector;

  cdf_Q2_nodes.push_back(0);
  cdf_vector.push_back(0);
  pdf_vector.push_back(0);

  // compute the spline table
  for (int i = 0; i < Q2spline.size(); ++i) {
      if (i == 0) {
          double cur_Q2 = Q2spline[i];
          double cur_pdf = normed_pdf(cur_Q2);
          double area = cur_Q2 * cur_pdf * 0.5;
          pdf_vector.push_back(cur_pdf);
          cdf_vector.push_back(area);
          cdf_Q2_nodes.push_back(cur_Q2);
          continue;
      }
      double cur_Q2 = Q2spline[i];
      double cur_pdf = normed_pdf(cur_Q2);
      double area = 0.5 * (pdf_vector[i - 1] + cur_pdf) * (Q2spline[i] - Q2spline[i - 1]);
      pdf_vector.push_back(cur_pdf);
      cdf_Q2_nodes.push_back(cur_Q2);
      cdf_vector.push_back(area + cdf_vector.back());
  }

  cdf_Q2_nodes.push_back(max);
  cdf_vector.push_back(1);
  pdf_vector.push_back(0);

  // set the spline table
  siren::utilities::TableData1D<double> inverse_cdf_data;
  inverse_cdf_data.x = cdf_vector;
  inverse_cdf_data.f = cdf_Q2_nodes;

  inverseCdf = siren::utilities::Interpolator1D<double>(inverse_cdf_data);
  return;

}

// this is temporary implementation
void CharmMesonDecay::SampleFinalStateHadronic(dataclasses::CrossSectionDistributionRecord & record, std::shared_ptr<siren::utilities::SIREN_random> random) const {
    std::vector<siren::dataclasses::SecondaryParticleRecord> & secondaries = record.GetSecondaryParticleRecords();
    siren::dataclasses::SecondaryParticleRecord & hadrons = secondaries[0];
    rk::P4 p4D_lab(geom3::Vector3(record.primary_momentum[1], record.primary_momentum[2], record.primary_momentum[3]), record.primary_mass);
    hadrons.SetFourMomentum({p4D_lab.e(), p4D_lab.px(), p4D_lab.py(), p4D_lab.pz()});
    hadrons.SetMass(p4D_lab.m());
    hadrons.SetHelicity(record.primary_helicity);
}

void CharmMesonDecay::SampleFinalState(dataclasses::CrossSectionDistributionRecord & record, std::shared_ptr<siren::utilities::SIREN_random> random) const {
    // first handle hadronic decay separately, in the end we want to handle all decays in separate functions
    dataclasses::InteractionSignature signature = record.signature;
    if (signature.secondary_types[0] == siren::dataclasses::Particle::ParticleType::Hadrons) {
      SampleFinalStateHadronic(record, random);
      return;
    }
    // first obtain the constants needed for further computation from the signature
    std::vector<double> constants = FormFactorFromRecord(record);
    double mD = particleMass(record.signature.primary_type);
    double mK = particleMass(record.signature.secondary_types[0]);

    // first sample a q^2
    double rand_value_for_Q2 = random->Uniform(0, 1);
    double Q2 = inverseCdf(rand_value_for_Q2);

    // now sample isotropically the "zenith" direction
    double cosTheta = random->Uniform(-1, 1);
    double sinTheta = std::sin(std::acos(cosTheta));
      // set the x axis to be the D direction
    geom3::UnitVector3 x_dir = geom3::UnitVector3::xAxis();
      //set the D direction in lab frame and compute its angle wrt the x axis
    rk::P4 p4D_lab(geom3::Vector3(record.primary_momentum[1], record.primary_momentum[2], record.primary_momentum[3]), record.primary_mass);
    geom3::Vector3 p3D_lab = p4D_lab.momentum();
    geom3::UnitVector3 p3D_lab_dir = p3D_lab.direction();
    geom3::Rotation3 x_to_p3D_lab_rot = geom3::rotationBetween(x_dir, p3D_lab_dir);
    // compute the momentum magnitude of the W and the K/pi
    double EK = 0.5 * (pow(mD, 2) + pow(mK, 2) - Q2) / mD; // energy of Kaon
    double PK = pow(pow(EK, 2) - pow(mK, 2), .5); // momentum magnitude of kaon in D rest frame
    double PW = sqrt(Q2); // momentum magnitude of virtual W in D rest frame
    // compute the 3 vectors of the W and the K/pi in the D rest frame, defined wrt x axis
    rk::P4 p4K_Drest(PK * geom3::Vector3(cosTheta, sinTheta, 0), mK);
    rk::P4 p4W_Drest(PK * geom3::Vector3(-cosTheta, -sinTheta, 0), PW); // invariant mass assigned to virtual W boson

    // rotate the momentum vectors so they are defined wrt to the D lab frame direction
    p4K_Drest.rotate(x_to_p3D_lab_rot);
    p4W_Drest.rotate(x_to_p3D_lab_rot);
    // perform the random "azimuth" rotation
    double phi = random->Uniform(0, 2 * M_PI);
    geom3::Rotation3 azimuth_rand_rot(p3D_lab_dir, phi);
    p4K_Drest.rotate(azimuth_rand_rot);
    p4W_Drest.rotate(azimuth_rand_rot);
    // finally, boost the 4 momenta back to the lab frame
    rk::Boost boost_from_Drest_to_lab = p4D_lab.labBoost();
    rk::P4 p4K_lab = p4K_Drest.boost(boost_from_Drest_to_lab);
    rk::P4 p4W_lab = p4W_Drest.boost(boost_from_Drest_to_lab);

    // this ends the computation of D->W+K/Pi decay, now treat the W->l+nu decay
    double ml = particleMass(record.signature.secondary_types[1]);
    double mnu = 0;
    double W_cosTheta = random->Uniform(-1, 1); // sampling the direction
    double W_sinTheta = std::sin(std::acos(W_cosTheta));
    double El = (Q2 + pow(ml, 2)) / (2 * sqrt(Q2));
    double Enu = (Q2 - pow(ml, 2)) / (2 * sqrt(Q2)); // the energies of the outgoing lepton and neutrino
    double P = (Q2 - pow(ml, 2)) / (2 * sqrt(Q2));
    // now we have thr four vectors of the outgoing particle kinematics in tne W rest frame wrt x direction
    rk::P4 p4l_Wrest(P * geom3::Vector3(W_cosTheta, W_sinTheta, 0), ml);
    rk::P4 p4nu_Wrest(P * geom3::Vector3(-W_cosTheta, -W_sinTheta, 0), 0);

    geom3::Vector3 p3W_lab = p4W_lab.momentum();
    geom3::UnitVector3 p3W_lab_dir = p3W_lab.direction();
    geom3::Rotation3 x_to_p3W_lab_rot = geom3::rotationBetween(x_dir, p3W_lab_dir);
    p4l_Wrest.rotate(x_to_p3W_lab_rot);
    p4nu_Wrest.rotate(x_to_p3W_lab_rot);
    // now finally perform the last aximuthal rotation
    double W_phi = random->Uniform(0, 2 * M_PI);
    geom3::Rotation3 W_azimuth_rand_rot(p3W_lab_dir, W_phi);
    p4l_Wrest.rotate(W_azimuth_rand_rot);
    p4nu_Wrest.rotate(W_azimuth_rand_rot);
    rk::Boost boost_from_Wrest_to_lab = p4W_lab.labBoost();
    rk::P4 p4l_lab = p4l_Wrest.boost(boost_from_Wrest_to_lab);
    rk::P4 p4nu_lab = p4nu_Wrest.boost(boost_from_Wrest_to_lab);

    std::vector<siren::dataclasses::SecondaryParticleRecord> & secondaries = record.GetSecondaryParticleRecords();
    siren::dataclasses::SecondaryParticleRecord & kpi = secondaries[0];
    siren::dataclasses::SecondaryParticleRecord & lepton = secondaries[1];
    siren::dataclasses::SecondaryParticleRecord & neutrino = secondaries[2]; //these are all hardcoded at the time

    kpi.SetFourMomentum({p4K_lab.e(), p4K_lab.px(), p4K_lab.py(), p4K_lab.pz()});
    kpi.SetMass(p4K_lab.m());
    kpi.SetHelicity(record.primary_helicity);

    lepton.SetFourMomentum({p4l_lab.e(), p4l_lab.px(), p4l_lab.py(), p4l_lab.pz()});
    lepton.SetMass(p4l_lab.m());
    lepton.SetHelicity(record.primary_helicity);

    neutrino.SetFourMomentum({p4nu_lab.e(), p4nu_lab.px(), p4nu_lab.py(), p4nu_lab.pz()});
    neutrino.SetMass(p4nu_lab.m());
    neutrino.SetHelicity(record.primary_helicity);

}

double CharmMesonDecay::FinalStateProbability(dataclasses::InteractionRecord const & record) const {
  double dd = DifferentialDecayWidth(record);
  double td = TotalDecayWidthForFinalState(record);
  if (dd == 0) return 0.;
  else if (td == 0) return 0.;
  else return dd/td;
}

std::vector<std::string> CharmMesonDecay::DensityVariables() const {
    return std::vector<std::string>{"Q2"};
}



} // namespace interactions
} // namespace siren

