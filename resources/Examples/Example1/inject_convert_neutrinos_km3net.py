#!/usr/bin/env python3
"""
KM3NeT SIREN Simulation Driver
==============================

This script drives the generation of neutrino events with the SIREN framework,
allowing flexible use of cross-section sets defined by subfolders in a root directory.

CONFIG FILE (as JSON via --config)
------------------------------------

Top-level keys (**required** unless noted):
- nev:           (int)     Number of events to generate.
- flavor:       (int)     Neutrino PDG code: 12 (nu_e), 14 (nu_mu), 16 (nu_tau), -12, -14, -16.
- interaction:   (str)     Either "all" (standard DIS) or "charm" (charm production).
- e_min:         (float)   Min neutrino energy (GeV).
- e_max:         (float)   Max neutrino energy (GeV).
- xs_root_dir:   (str)     Root directory where cross-section subfolders live.
- xs_subdir:     (str)     Subfolder inside xs_root_dir for this process.
                          E.g., "M_Muon_New", "M_Tau", "CSMS", "CSMS_v1.0".
- experiment:    (str)     [Optional] SIREN experiment label (default: "KM3NeTORCA").
- gamma:         (float)   [Optional] Power-law index for flux (default: 2.0).
- seed:          (int)     [Optional, CLI overrides] Random seed.

Additional required for "charm" interaction:
- current_type:  (str)     "cc", "nc" or "both" (which currents to simulate).

Filenames convention:
- For xs_subdir == "M_Muon", "M_Tau", etc:
    - 'all':   dsdxdy_nu-N-cc-HERAPDF20_NLO_EIG_central.fits (etc)
    - 'charm': dsdxdy_nu-N-cc-charm-EPPS21nlo_CT18Anlo_O16_central.fits, etc.
- For xs_subdir == "CSMS", "CSMS_v1.0":
    - dsdxdy_nu_CC_iso.fits, sigma_nu_CC_iso.fits, dsdxdy_nu_NC_iso.fits, sigma_nu_NC_iso.fits

EXAMPLES:
---------

# Standard DIS (M_Muon_New set)
{
  "nev": 10000,
  "flavor": 14,
  "e_min": 1000,
  "e_max": 1000000,
  "interaction": "all",
  "xs_root_dir": "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/DBSearch/Simulation/Resources/Splines",
  "xs_subdir": "M_Muon_New",
  "seed": 42
}

# Charm production (both CC and NC, for tau)
{
  "nev": 5000,
  "flavor": 16,
  "e_min": 1000,
  "e_max": 1000000,
  "interaction": "charm",
  "current_type": "both",
  "xs_root_dir": "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/DBSearch/Simulation/Resources/Splines",
  "xs_subdir": "M_Tau",
  "seed": 314
}

# DIS using CSMS set
{
  "nev": 5000,
  "flavor": 14,
  "e_min": 1000,
  "e_max": 1000000,
  "interaction": "all",
  "xs_root_dir": "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/DBSearch/Simulation/Resources/Splines",
  "xs_subdir": "CSMS"
}
"""

import argparse
import os
import json
import siren
from siren.SIREN_Controller import SIREN_Controller
import sys 
#print location of SIREN being used
print("SIREN path:", siren.__path__[0])
import awkward as ak
import numpy as np
import ROOT
from tqdm import tqdm
import aa
import sys
import numpy as np
from math import modf

namecodes = {
     12 : 'nu',
    -12 : 'nubar',
     14 : 'nu',
    -14 : 'nubar',
     16 : 'nu',
    -16 : 'nubar',
}

SIREN_PARTICLES = {
    12: siren.dataclasses.Particle.ParticleType.NuE,
    14: siren.dataclasses.Particle.ParticleType.NuMu,
    16: siren.dataclasses.Particle.ParticleType.NuTau,
    -12: siren.dataclasses.Particle.ParticleType.NuEBar,
    -14: siren.dataclasses.Particle.ParticleType.NuMuBar,
    -16: siren.dataclasses.Particle.ParticleType.NuTauBar
}

c = 3e-1 # m / ns
MJD0 = 40587. # MJD corresponding to 01.01.1970 00:00:00 UTC

namecodes = {
     12 : 'nu',
    -12 : 'nubar',
     14 : 'nu',
    -14 : 'nubar',
     16 : 'nu',
    -16 : 'nubar',
}

def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def get_xs_prepend(config) -> str:
    """Return the full path to the relevant cross-section subfolder."""
    return os.path.join(config["xs_root_dir"], config["xs_subdir"])

def get_cross_section_paths(config, xs_prepend, mode):
    """
    Return a dict {'cc': [...], 'nc': [...]} where
    each value is a list of (dsdy, sigma) tuples, for all available targets.

    - "all" mode, CSMS: 1 pair per group, CSMS iso naming
    - "all" mode, standard: 1 pair per group, HERAPDF naming
    - "charm" mode, CSMS: only oxygen, but uses CC/NC iso names
    - "charm" mode, myCharmNew: both O16 and H, full set
    - "charm" mode, other: only hydrogen
    """
    is_csms = config["xs_subdir"].startswith("CSMS")
    mode = mode.lower()

    if mode == "all":
        if is_csms:
            cc_paths = [(
                os.path.join(xs_prepend, "dsdxdy_nu_CC_iso.fits"),
                os.path.join(xs_prepend, "sigma_nu_CC_iso.fits")
            )]
            nc_paths = [(
                os.path.join(xs_prepend, "dsdxdy_nu_NC_iso.fits"),
                os.path.join(xs_prepend, "sigma_nu_NC_iso.fits")
            )]
        else:
            cc_paths = [(
                os.path.join(xs_prepend, "dsdxdy_nu-N-cc-HERAPDF20_NLO_EIG_central.fits"),
                os.path.join(xs_prepend, "sigma_nu-N-cc-HERAPDF20_NLO_EIG_central.fits")
            )]
            nc_paths = [(
                os.path.join(xs_prepend, "dsdxdy_nu-N-nc-HERAPDF20_NLO_EIG_central.fits"),
                os.path.join(xs_prepend, "sigma_nu-N-nc-HERAPDF20_NLO_EIG_central.fits")
            )]
        return {'cc': cc_paths, 'nc': nc_paths}

    elif mode == "charm":
        if config["charm_file"] is None:
            raise ValueError("For 'charm' interaction, 'charm_file' must be specified in config.")
        is_new = (config["charm_file"] == "myCharmNew")
        oxygen_pdf = "EPPS21nlo_CT18Anlo_O16_central"
        hydrogen_pdf = "HERAPDF20_NLO_EIG_central"
        cc_paths = []
        nc_paths = []

        if is_csms:
            # CSMS mode for charm: use CC/NC iso, oxygen only (guess)
            cc_paths.append((
                os.path.join(xs_prepend, "dsdxdy_nu_CC_iso.fits"),
                os.path.join(xs_prepend, "sigma_nu_CC_iso.fits")
            ))
            nc_paths.append((
                os.path.join(xs_prepend, "dsdxdy_nu_NC_iso.fits"),
                os.path.join(xs_prepend, "sigma_nu_NC_iso.fits")
            ))
        elif is_new:
            # myCharmNew: FULL SET (O16 and H)
            cc_paths.extend([
                (os.path.join(xs_prepend, f"dsdxdy_nu-N-cc-charm-{oxygen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_nu-N-cc-charm-{oxygen_pdf}.fits")),
                (os.path.join(xs_prepend, f"dsdxdy_nu-N-cc-charm-{hydrogen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_nu-N-cc-charm-{hydrogen_pdf}.fits"))
            ])
            nc_paths.extend([
                (os.path.join(xs_prepend, f"dsdxdy_nu-N-nc-charm-{oxygen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_nu-N-nc-charm-{oxygen_pdf}.fits")),
                (os.path.join(xs_prepend, f"dsdxdy_nu-N-nc-charm-{hydrogen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_nu-N-nc-charm-{hydrogen_pdf}.fits"))
            ])
        else:
            # Not new: Only hydrogen, as in legacy
            cc_paths.append((
                os.path.join(xs_prepend, f"dsdxdy_nu-N-cc-charm-{hydrogen_pdf}.fits"),
                os.path.join(xs_prepend, f"sigma_nu-N-cc-charm-{hydrogen_pdf}.fits")
            ))
            nc_paths.append((
                os.path.join(xs_prepend, f"dsdxdy_nu-N-nc-charm-{hydrogen_pdf}.fits"),
                os.path.join(xs_prepend, f"sigma_nu-N-nc-charm-{hydrogen_pdf}.fits")
            ))
        return {'cc': cc_paths, 'nc': nc_paths}

    else:
        raise ValueError(f"Unknown mode '{mode}' for cross section path selection.")

def configure_standard_dis_cross_sections(config, primary_type, target_type):
    """
    Create a DIS (all-channel) InteractionCollection from cross-section paths
    using the new grouped-by-cc/nc get_cross_section_paths output.

    Adds only CC or NC as requested by config["current_type"] ("cc", "nc", or "both").
    """
    xs_prepend = get_xs_prepend(config)
    paths = get_cross_section_paths(config, xs_prepend, mode="all")
    print(f"[INFO] Using cross-section paths:\n CC: {paths['cc']}\n NC: {paths['nc']}")
    isoscalar_mass = float(0.938272 + 0.939565) / 2
    xs_list = []
    current_type = config.get("current_type", "both").lower()  # default: include both

    # cc interactions
    if current_type in ("cc", "both"):
        for dsdy, sigma in paths['cc']:
            xs_list.append(
                siren.interactions.DISFromSpline(
                    dsdy, sigma, 1, isoscalar_mass, 0.01, [primary_type], [target_type], "m"
                )
            )
    # nc interactions
    if current_type in ("nc", "both"):
        for dsdy, sigma in paths['nc']:
            xs_list.append(
                siren.interactions.DISFromSpline(
                    dsdy, sigma, 2, isoscalar_mass, 1.0, [primary_type], [target_type], "m"
                )
            )
    return siren.interactions.InteractionCollection(primary_type, xs_list)

def configure_charm_cross_sections(config, primary_type):
    """
    Create a charm InteractionCollection given new get_cross_section_paths logic.
    The targets are mapped in the order [O16, H] for charm, [H] only for old.
    """
    xs_prepend = get_xs_prepend(config)
    paths = get_cross_section_paths(config, xs_prepend, mode="charm")
    isoscalar_mass = float(0.938272 + 0.939565) / 2
    current_type = config["current_type"].lower()
    print(f"[INFO] Using cross-section paths:\n CC: {paths['cc']}\n NC: {paths['nc']}")
    # O16 first, then H (legacy), or just H for old mode
    targets = []
    if len(paths['cc']) == 2:  # True if both O16/H; False if just H.
        targets = [
            siren.dataclasses.Particle.ParticleType.O16Nucleus, 
            siren.dataclasses.Particle.ParticleType.HNucleus
        ]
    else:
        targets = [siren.dataclasses.Particle.ParticleType.HNucleus]

    xs_list = []
    if current_type in ("cc", "both"):
        for idx, (dsdy, sigma) in enumerate(paths['cc']):
            xs_list.append(
                siren.interactions.QuarkDISFromSpline(
                    dsdy, sigma, 1, isoscalar_mass, 1,
                    [primary_type], [targets[idx]], "m"
                )
            )
    if current_type in ("nc", "both"):
        for idx, (dsdy, sigma) in enumerate(paths['nc']):
            xs_list.append(
                siren.interactions.QuarkDISFromSpline(
                    dsdy, sigma, 2, isoscalar_mass, 1,
                    [primary_type], [targets[idx]], "m"
                )
            )
    return siren.interactions.InteractionCollection(primary_type, xs_list)

def build_distributions(config, controller, primary_type):
    
    """Build SIREN injection and physical distributions."""
    
    p_inj, p_phys = {}, {}
    #set mass of primary neutrinos 
    mass_dist = siren.distributions.PrimaryMass(0)
    p_inj["mass"] = mass_dist
    p_phys["mass"] = mass_dist
    
    #energy distribution
    e_min, e_max = float(config["emin_gev"]), float(config["emax_gev"])
    gamma = config["gamma"]
    edist = siren.distributions.PowerLaw(gamma, e_min, e_max)
    
    p_phys["energy"] = edist
    p_inj["energy"] = edist
    #direction distribution 
    direction_distribution = siren.distributions.IsotropicDirection()
    p_inj["direction"] = direction_distribution
    p_phys["direction"] = direction_distribution
    
    #position distribution
    muon_range_func = siren.distributions.LeptonDepthFunction()
    position_distribution = siren.distributions.ColumnDepthPositionDistribution(
        600, 600.0, muon_range_func)
    
    p_inj["position"] = position_distribution
    
    return p_inj, p_phys

def add_secondary_to_controller(controller, secondary_type, secondary_xsecs, secondary_decays = None):
    
    if secondary_decays is not None:
        secondary_collection = siren.interactions.InteractionCollection(secondary_type, [secondary_xsecs], [secondary_decays])
    else:
        secondary_collection = siren.interactions.InteractionCollection(secondary_type, [secondary_xsecs])
    
    secondary_injection_process = siren.injection.SecondaryInjectionProcess()
    secondary_physical_process = siren.injection.PhysicalProcess()
    secondary_injection_process.primary_type = secondary_type
    secondary_physical_process.primary_type = secondary_type
    secondary_injection_process.AddSecondaryInjectionDistribution(siren.distributions.SecondaryPhysicalVertexDistribution())
    controller.secondary_injection_processes.append(secondary_injection_process)
    controller.secondary_physical_processes.append(secondary_physical_process)

    return secondary_collection

def run_simulation(config: dict,detname: str, output: str, seed: int):
    """Run SIREN simulation with correct cross-section paths."""
    
    events_to_inject = int(float(config["nev"]))
    nu_flavor = config["flavor"]
    interaction = config["interaction"].lower()
    
    if nu_flavor not in SIREN_PARTICLES:
        raise ValueError(f"Flavor code {nu_flavor} is not valid.")
    
    primary_type = SIREN_PARTICLES[nu_flavor]
    print(f"[INFO] Primary particle type: {primary_type.name}")
    
    #experiment = config.get("experiment", "KM3NeTORCA")
    
    target_type = siren.dataclasses.Particle.ParticleType.Nucleon
    print(f"This is the file I am getting the experiment densities from: {detname}")
    controller = SIREN_Controller(events_to_inject, detname, seed=seed)
    
    if interaction == "charm":
        print("Configuring for CHARM production.")
        primary_xs = configure_charm_cross_sections(config, primary_type)
    elif interaction == "all":
        print("Configuring for STANDARD DIS.")
        primary_xs = configure_standard_dis_cross_sections(config, primary_type, target_type)
    else:
        raise ValueError(f"Invalid interaction type: {interaction}")
    
    controller.SetInteractions(primary_xs)
    inj_distrib, phys_distrib = build_distributions(config, controller, primary_type)
    controller.SetProcesses(primary_type, inj_distrib, phys_distrib)
    
    if interaction == "charm":
        DPlus  = siren.dataclasses.Particle.ParticleType.DPlus
        D0     = siren.dataclasses.Particle.ParticleType.D0
        D0Bar  = siren.dataclasses.Particle.ParticleType.D0Bar
        DMinus = siren.dataclasses.Particle.ParticleType.DMinus
        force_muonic = (config.get("decay", "") == "muonic")
        DPlus_decay  = siren.interactions.CharmMesonDecay(primary_type=DPlus,  force_muonic=force_muonic)
        D0_decay     = siren.interactions.CharmMesonDecay(primary_type=D0,     force_muonic=force_muonic)
        D0Bar_decay  = siren.interactions.CharmMesonDecay(primary_type=D0Bar,  force_muonic=force_muonic)
        DMinus_decay = siren.interactions.CharmMesonDecay(primary_type=DMinus, force_muonic=force_muonic)
        D_energy_loss = siren.interactions.DMesonELoss()
        secondary_DPlus_collection  = add_secondary_to_controller(controller, DPlus,  D_energy_loss, DPlus_decay)
        secondary_D0_collection     = add_secondary_to_controller(controller, D0,     D_energy_loss, D0_decay)
        secondary_D0Bar_collection  = add_secondary_to_controller(controller, D0Bar,  D_energy_loss, D0Bar_decay)
        secondary_DMinus_collection = add_secondary_to_controller(controller, DMinus, D_energy_loss, DMinus_decay)
        controller.SetInteractions(primary_xs, [secondary_DPlus_collection, secondary_D0_collection, secondary_D0Bar_collection, secondary_DMinus_collection])
        print("This is the controller after adding charm secondaries: ", controller)
    controller.Initialize()
    def stop(datum, i):
        return False
    controller.SetInjectorStoppingCondition(stop)
    controller.GenerateEvents()
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    complete_path = os.path.abspath(output)
    print("Complete path before seed:", complete_path)
    #remove last .parquet path of the path if exists
    
    if complete_path.endswith(".parquet"):
            complete_path = complete_path[:-8]
    print("Saving events to:", complete_path)
    
    controller.SaveEvents(complete_path)
    return complete_path

class Particle:
    def __init__(self, pdg, momentum, position, time, mother_id=None, status=None, energy=None, name=''):
        self.pdg = pdg                      # PDG code
        self.momentum = momentum            # 3-vector, e.g., [px, py, pz] or np.array
        self.position = position            # 3-vector, e.g., [x, y, z]
        self.time = time                    # float
        self.mother_id = mother_id          # which particle made me?
        self.status = status                # optional, e.g. 1=final, 2=intermediate
        self.energy = energy                # optional, if you want
        self.name = name                    # for convenience/debugging

    def __repr__(self):
        return f"Particle({self.pdg}, {self.momentum}, {self.position}, {self.time})"
        
class Interaction:
    def __init__(self, parent, daughters, intermediates=None, event_id=None, meta=None):
        self.parent = parent                  # Particle (single)
        self.daughters = list(daughters)      # list of Particle
        self.intermediates = intermediates or [] # Optionally: intermediate Particle list
        self.meta = meta or {}                # Optional: dict for weights, other info

    def final_state_particles(self):
        """Return the final-state particles (daughters)."""
        return self.daughters

    def __repr__(self):
        return (f"Interaction(parent={self.parent}, "
                f"daughters={self.daughters}, "
                )

def create_interaction(event, time):
    parent = Particle(
        pdg=event["primary_type"][0],
        momentum=event["primary_momentum"][0][1:],
        position=event["vertex"][0],
        time = time,
        mother_id=-1,
        status=0, # primary
        energy=event["primary_momentum"][0][0],
        name="primary"
    )
    
    daughters = []
    intermediates = []
    #Only keep final state particles 
    #primary types 
    primary_types = event["primary_type"].to_list()
    secondary_types = event["secondary_types"].to_list()
    secondary_momenta = event["secondary_momenta"].to_list()
    flattened_secondary_types = [item for sublist in secondary_types for item in sublist]
    flattened_secondary_momenta = [item for sublist in secondary_momenta for item in sublist]
    
    for pdg, mom in zip(flattened_secondary_types, flattened_secondary_momenta):
        
        
        if pdg in primary_types:
            intermediate = Particle(
                pdg=pdg,
                momentum=mom[1:],
                position=event["vertex"][0],
                time=time,
                mother_id=1, # from primary
                status=2, # intermediate
                energy=mom[0],
                name="intermediate"
            )
            intermediates.append(intermediate)
        else:
            if ROOT.TMath.Abs(pdg)>1000000000: #TODO check if this has to be included
                pdg = 211
            daughter = Particle(
                pdg=pdg,
                momentum=mom[1:],
                position=event["vertex"][0],
                time=time,
                mother_id=1, # from primary
                status=1, # final state
                energy=mom[0],
                name="daughter"
            )
            daughters.append(daughter)
    
    interaction = Interaction(
        parent=parent,
        daughters=daughters,
        intermediates=intermediates,
        meta={"event_weight": event["event_weight"]}
    )
    return interaction
    
def interaction_to_root_event(interaction: Interaction, **kwargs):
    
    idx = kwargs["idx"]
    options = kwargs["options"]
    nevts = options.n
    ene_factor = kwargs["ene_factor"]
    #Time information
    TimeStampStart = int(float(options.t.split(",")[0]))
    TimeStampStop  = int(float(options.t.split(",")[1]))
    MJDStart = CalcMJDFromTimeStamp(TimeStampStart, 0)
    MJDStop  = CalcMJDFromTimeStamp(TimeStampStop, 0)
    can1,can2,can3  = float(options.c.split(",")[0]),float(options.c.split(",")[1]),float(options.c.split(",")[2])
    #Detector geometry
    origin = ROOT.Vec(float(options.x.split(",")[0]), float(options.x.split(",")[1]), (can2-can1)/2)
    
    alpha = options.a
    RNG = np.random.default_rng( seed = options.r )
    #Create the ROOT event
    Revt = ROOT.Evt()
    Revt.mc_run_id = options.r # run id
    Revt.mc_id = idx 
    Revt.id = idx
    Revt.mc_t = 0
    
    fMJD = RNG.uniform(MJDStart,MJDStop)
    TimeStampSec, TimeStampTick = CalcTimeStamp(fMJD)
    
    Revt.mc_event_time.SetSec(TimeStampSec)
    Revt.mc_event_time.SetNanoSec(TimeStampTick*16)
    
    #first the incoming neutrino 
    parent = interaction.parent
    pos1 = ROOT.Vec(parent.position[0],parent.position[1],parent.position[2])
    pos1 += origin
    
    trk = ROOT.Trk()
    trk.id = 0
    trk.mother_id = parent.mother_id
    trk.pos = pos1
    trk.dir = ROOT.Vec(parent.momentum[0]/np.linalg.norm(parent.momentum[1:]), 
                        parent.momentum[1]/np.linalg.norm(parent.momentum[1:]),
                        parent.momentum[2]/np.linalg.norm(parent.momentum[1:]))
    trk.E = parent.energy
    trk.type = int(parent.pdg)
    trk.t = 0
    trk.status =  100 #TODO is this the correct status for primary?
    Revt.mc_trks.push_back( trk )
    
    Faanet  = ROOT.Flux_Atmospheric()
    #then loop through the daughter particles 
    for i, daughter in enumerate(interaction.daughters):
        trk = ROOT.Trk()
        trk.id = i+1
        trk.mother_id = daughter.mother_id
        #add origin to the position
        pos2 = ROOT.Vec(daughter.position[0], daughter.position[1], daughter.position[2])
        pos2 += origin
        trk.pos = pos2
        trk.dir = ROOT.Vec(daughter.momentum[0]/np.linalg.norm(daughter.momentum[:]), 
                            daughter.momentum[1]/np.linalg.norm(daughter.momentum[:]),
                            daughter.momentum[2]/np.linalg.norm(daughter.momentum[:]))
        trk.E = daughter.energy
        trk.type = int(daughter.pdg)
        trk.t = 0
        trk.status =  1 #TODO is this the correct status for final status of the particles ? 
        Revt.mc_trks.push_back( trk )
    
    oneweight = interaction.meta["event_weight"]*nevts*ene_factor*ROOT.TMath.Power(Revt.mc_trks[0].E,alpha)*4*ROOT.TMath.Pi()
    
    faanet = Faanet.dNdEdOmega(Revt.mc_trks[0].type,ROOT.TMath.Log10(Revt.mc_trks[0].E),Revt.mc_trks[0].dir.z)
    
    Revt.w = [interaction.meta["event_weight"],oneweight,faanet*oneweight]
    
    return Revt

def CalcMJDFromTimeStamp(TimeStampSec, TimeStampTick) :
    # computes MJD from the TimeStamp as number of seconds counted since
    # 01.01.1970 00:00:00  UTC and 16 nanosecond-ticks.
    MJD = TimeStampSec + TimeStampTick/62500000
    MJD /= 24.*3600.
    MJD += MJD0
    return MJD

def CalcTimeStamp(MJD) :
    # computes the TimeStamp as number of seconds counted since 01.01.1970
    # 00:00:00  UTC and 16 nanosecond-ticks.
    MJD0 = 40587 # MJD corresponding to 01.01.1970 00:00:00 UTC
    PartFrac,PartInt = modf(((MJD - MJD0) * 24. * 3600.))
    PartFrac *= 62500000
    return int(PartInt),int(PartFrac)

def write_gSeaGen_rootfile(siren_filename,options):
    
    if not siren_filename.endswith(".parquet"):
        raise ValueError("Input file is not a parquet file")
    
    data= ak.from_parquet(siren_filename)
    nevts = len(data)
    
    if nevts!=options.n :
        print('Error: Different input ngen and number of generated events in sirene',options.n,nevts) #Does this have any effect on the output ?
        #exit()

    TimeStampStart = int(float(options.t.split(",")[0]))
    TimeStampStop  = int(float(options.t.split(",")[1]))
    
    if siren_filename.endswith(".parquet"):
        siren_filename = siren_filename[:-8]
        complete_path = siren_filename + ".root"
    else:
        ValueError("Input file is not a parquet file")
    
    can1,can2,can3  = float(options.c.split(",")[0]),float(options.c.split(",")[1]),float(options.c.split(",")[2])
    outfile = ROOT.OutputEventFile(complete_path)
    #Write Header to the ROOT file 
    header = ROOT.Head()
    header.set_line("start_run",     str(options.r) )
    header.set_line("seed",          str(options.r) )
    #header.set_line("simul",         'siren '+siren.__path__[0]+' '+str(np.datetime64('now')))
    header.set_line("cut_ene",       str(options.e)+" "+str(options.me))
    header.set_line("alpha",         str(options.a))
    header.set_line("nupdg",         str(options.f))
    header.set_line("model",         str(options.M))
    header.set_line("fixedcan",      str(options.x.replace(","," "))+" "+str(options.c.replace(","," ")) )
    header.set_line("depth",         str(options.d) )
    header.set_line("time_interval", str(TimeStampStart)+" "+str(TimeStampStop))
    header.set_line("simul", "gSeaGen") #Check if this is correct for the implementation to work 
    intvol = ROOT.TMath.Pi()*can3*can3*(can2-can1)*1.03975 #density of seawater
    header.set_line("genvol", str(can1)+" "+str(can2)+" "+str(can3)+" "+str(intvol)+" "+str(nevts))
    print( "Header: \n", header )

    outfile.write_header(header)
    
    # Fnuflux = nuflux.makeFlux('H3a_SIBYLL23C')
    # Fdaemon = daemonflux.Flux(location="generic", use_calibration=True, debug=1)

    alpha      = options.a
    ene_factor = (ROOT.TMath.Power(options.me,-alpha+1)-ROOT.TMath.Power(options.e,-alpha+1))/(-alpha+1)
    #Write event of the root file
    for idx, evt in tqdm(enumerate(data), total=len(data)):
        interaction = create_interaction( evt, time=0 )
        R_evt = interaction_to_root_event( interaction, options=options, idx=idx, ene_factor=ene_factor )
        outfile.write( R_evt )
    
    del outfile
    sys.exit(0)
def read_can_txt(filename):
    """
    Reads a txt file with CAN_*_M=... lines and returns a dict of values as floats.
    """
    variables = {}
    with open(filename) as f:
        for line in f:
            # Strip whitespace and ignore empty lines
            line = line.strip()
            if not line or '=' not in line:
                continue
            # Partition on the first "="
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Try to store as float if possible, otherwise as string
            try:
                value = float(value)
            except ValueError:
                pass
            variables[key] = value
    return variables

def CreateTempSirenDetector(detname,height,radius,depth) :
    print(f"detname:", detname)
    print(f"height", height)
    print(f"radius:", radius)
    print(f"depth", depth)
    # copies the defualt detector file defined in SIREN and creates
    # a new one based on the configuration of a given run
    detdir = siren.__path__[0]+'/resources/Detectors'

    print('rm -fr '+detdir+'/materials/'+detname)
    print('rm -fr '+detdir+'/densities/'+detname)
    print('cp -r '+detdir+'/materials/KM3NeTORCA '+detdir+'/materials/'+detname)
    print('cp -r '+detdir+'/densities/KM3NeTORCA '+detdir+'/densities/'+detname)
    print('mv '+detdir+'/materials/'+detname+'/KM3NeTORCA-v1.dat '+detdir+'/materials/'+detname+'/'+detname+'-v1.dat')
    print('mv '+detdir+'/densities/'+detname+'/KM3NeTORCA-v1.dat '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')
    print('sed -i \'s/120/'+str(radius)+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')
    print('sed -i \'s/220/'+str(height)+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')
    print('sed -i \'s/6368110/'+str(6368000+height/2)+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')
    print('sed -i \'s/6370440/'+str(int(6368000+depth))+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')

    os.system('rm -rf '+detdir+'/materials/'+detname)
    os.system('rm -rf '+detdir+'/densities/'+detname)

    os.system('cp -r '+detdir+'/materials/KM3NeTORCA '+detdir+'/materials/'+detname)
    os.system('cp -r '+detdir+'/densities/KM3NeTORCA '+detdir+'/densities/'+detname)

    os.system('mv '+detdir+'/materials/'+detname+'/KM3NeTORCA-v1.dat '+detdir+'/materials/'+detname+'/'+detname+'-v1.dat')
    os.system('mv '+detdir+'/densities/'+detname+'/KM3NeTORCA-v1.dat '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')

    os.system('sed -i \'s/120/'+str(radius)+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')
    os.system('sed -i \'s/220/'+str(height)+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')
    os.system('sed -i \'s/6368110/'+str(6368000+height/2)+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')
    os.system('sed -i \'s/6370440/'+str(int(6368000+depth))+'/g\' '+detdir+'/densities/'+detname+'/'+detname+'-v1.dat')

    # also create the SIREN physics-model dir that SIREN_Controller searches
    detectors_dir = siren.__path__[0]+'/resources/detectors'
    os.system('rm -rf '+detectors_dir+'/'+detname)
    os.system('cp -r '+detectors_dir+'/KM3NeTORCA '+detectors_dir+'/'+detname)
    
    return detdir

def main():
    """
    options:
    -i     : The input file                       default=""
    -o     : The output file, aanet format        default=""
    -c     : detector can                         default=0,0,0
    -x     : detector origin                      default=0,0
    -d     : detector depth                       default=0.
    -r     : run id and seed                      default=1
    -n     : generate this many events            default=1
    -a     : Energy spectrum                      default=0.
    -e     : Emin / GeV                           default=10.
    -me    : Emax / GeV                           default=100.
    -f     : Pdg code input neutrino              default=14
    -t     : time stampt                          default="0.,100."
    -M    : model name                            default=""
    """
    
    parser = argparse.ArgumentParser(
        description="Run KM3NeT SIREN Simulation with explicit subfolder cross-section path logic.")
    parser.add_argument('--config', required=True, type=json.loads,
                        help="Settings dictionary as JSON string.")
    parser.add_argument('--output', type=str, default="output", help="Output directory")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--detector_info', type=str, help="Detector file with information in position and can")
    
    args = parser.parse_args()
    detector_file = args.detector_info
    
    can_info = read_can_txt(detector_file)
    
    can1,can2,can3  = float(can_info['CAN_Zmin_M']),float(can_info['CAN_Zmax_M']),float(can_info['CAN_Radius_M'])
    depth = float(can_info['CAN_Depth_M'])
    #detsufix = namecodes[options.f]+str(options.r)
    
    detsufix = namecodes[args.config['flavor']]+str(args.seed)
    detdir = CreateTempSirenDetector(detsufix,can2-can1,can3,depth)
    
    print(f"THis is the detsuffix that i am using: {detsufix}")
    path = run_simulation(config = args.config,
                          detname= detsufix, 
                          output = args.output, 
                          seed = args.seed)
    
    parquet_path = path + ".parquet"
    root_path = path + ".root"
    
    myopts = {
        "-i": parquet_path,
        "-o": root_path,
        "-c": f"{can_info['CAN_Zmin_M']},{can_info['CAN_Zmax_M']},{can_info['CAN_Radius_M']}",
        "-x": f"{can_info['CAN_X_M']},{can_info['CAN_Y_M']}",
        "-d": f"{can_info['CAN_Depth_M']}", #should this be + or - ? 
        "-r": "1",
        "-n": str(int(float(args.config['nev']))), #takes care of 1e4 notation
        "-a": f"{args.config['gamma']}",
        "-e": f"{args.config['emin_gev']}",
        "-me": f"{args.config['emax_gev']}",
        "-f": f"{args.config['flavor']}",
        "-t": "0.,100.",
        "-M": "gSeaGen"
    }
    # Convert dictionary to argument list
    argv = [str(x) for kv in myopts.items() for x in kv]
    options = aa.Options(main.__doc__, argv)
    
    print("\n[INFO] Using config: \n" + json.dumps(args.config, indent=2))
    
    print(f"detsufix", detsufix)

    #get options config from the file that does the injections
    print(f" can info: {can1}, {can2}, {can3}, depth: {depth}")
    
    write_gSeaGen_rootfile(parquet_path, options=options) # <--- CORRECT! 
    
    #os.system('rm -f '+sirenfilename+'.*')
    os.system('rm -fr '+detdir+'/materials/'+detsufix)
    os.system('rm -fr '+detdir+'/densities/'+detsufix)
    
if __name__ == "__main__":
    main()