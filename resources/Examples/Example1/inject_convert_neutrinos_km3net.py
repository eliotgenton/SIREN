#!/usr/bin/env python3
"""
KM3NeT SIREN Simulation Driver  — modern siren.injection.Injector API
======================================================================

Replaces the SIREN_Controller-based version (moved to legacy/ 2026-05-05).
Uses the upstream PR #71 Injector/Weighter idiom which fixes the
multi-event multi-target crash in SIREN_Controller.SetProcesses.

CONFIG FILE (as JSON via --config)
------------------------------------

Top-level keys (**required** unless noted):
- nev:           (int)     Number of events to generate.
- flavor:        (int)     Neutrino PDG code: 12 (nu_e), 14 (nu_mu), 16 (nu_tau), -12, -14, -16.
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

EXAMPLES:
---------

# Standard DIS (M_Muon_New set)
{
  "nev": 10000,
  "flavor": 14,
  "e_min": 1000,
  "e_max": 1000000,
  "interaction": "all",
  "xs_root_dir": "/pbs/home/e/egenton/SIREN/resources/CrossSections/all_cross_sections",
  "xs_subdir": "M_Muon_New",
  "seed": 42
}

# Charm production (both CC and NC, forced muonic D-decay)
{
  "nev": 5000,
  "flavor": 14,
  "e_min": 100,
  "e_max": 10000,
  "interaction": "charm",
  "current_type": "both",
  "xs_root_dir": "/pbs/home/e/egenton/SIREN/resources/CrossSections/all_cross_sections",
  "xs_subdir": "M_Muon_New",
  "charm_file": "myCharmNew",
  "decay": "muonic",
  "seed": 314
}
"""

import argparse
import os
import json
import sys
import numpy as np
from math import modf

import siren
from siren._util import GenerateEvents, SaveEvents

print("SIREN path:", siren.__path__[0])

import awkward as ak

# ROOT, tqdm, and aa are only needed by write_gSeaGen_rootfile / main.
# Deferred so run_simulation() can be imported and tested without ROOT
# (the SIREN Python env intercepts bare `import ROOT` before aanet sets
# up the ROOT library path, causing ImportError if ROOT is at module level).

# ---------------------------------------------------------------------------
# Constants / mappings
# ---------------------------------------------------------------------------

namecodes = {
     12 : 'nu',
    -12 : 'nubar',
     14 : 'nu',
    -14 : 'nubar',
     16 : 'nu',
    -16 : 'nubar',
}

SIREN_PARTICLES = {
    12: siren.dataclasses.ParticleType.NuE,
    14: siren.dataclasses.ParticleType.NuMu,
    16: siren.dataclasses.ParticleType.NuTau,
    -12: siren.dataclasses.ParticleType.NuEBar,
    -14: siren.dataclasses.ParticleType.NuMuBar,
    -16: siren.dataclasses.ParticleType.NuTauBar,
}

# Branching ratios for D-meson -> K mu nu (PDG values)
# Used to correct event weights when forcing muonic decay
BR_MUONIC = {411: 0.176, -411: 0.176, 421: 0.067, -421: 0.067}

c    = 3e-1   # m / ns
MJD0 = 40587. # MJD corresponding to 01.01.1970 00:00:00 UTC

# ---------------------------------------------------------------------------
# Cross-section helpers  (unchanged from legacy — pure path logic)
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_xs_prepend(config) -> str:
    return os.path.join(config["xs_root_dir"], config["xs_subdir"])

def get_cross_section_paths(config, xs_prepend, mode):
    """
    Return a dict {'cc': [...], 'nc': [...]} where each value is a list of
    (dsdy, sigma) tuples.  Identical logic to the legacy version.
    """
    is_csms = config["xs_subdir"].startswith("CSMS")
    mode    = mode.lower()
    flavor  = int(config["flavor"])
    sign    = "nu" if flavor > 0 else "nubar"

    if mode == "all":
        if is_csms:
            cc_paths = [(
                os.path.join(xs_prepend, f"dsdxdy_{sign}_CC_iso.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}_CC_iso.fits")
            )]
            nc_paths = [(
                os.path.join(xs_prepend, f"dsdxdy_{sign}_NC_iso.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}_NC_iso.fits")
            )]
        else:
            cc_paths = [(
                os.path.join(xs_prepend, f"dsdxdy_{sign}-N-cc-HERAPDF20_NLO_EIG_central.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}-N-cc-HERAPDF20_NLO_EIG_central.fits")
            )]
            nc_paths = [(
                os.path.join(xs_prepend, f"dsdxdy_{sign}-N-nc-HERAPDF20_NLO_EIG_central.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}-N-nc-HERAPDF20_NLO_EIG_central.fits")
            )]
        return {'cc': cc_paths, 'nc': nc_paths}

    elif mode == "charm":
        if config.get("charm_file") is None:
            raise ValueError("For 'charm' interaction, 'charm_file' must be specified in config.")
        is_new        = (config["charm_file"] == "myCharmNew")
        oxygen_pdf    = "EPPS21nlo_CT18Anlo_O16_central"
        hydrogen_pdf  = "HERAPDF20_NLO_EIG_central"
        cc_paths, nc_paths = [], []

        if is_csms:
            cc_paths.append((
                os.path.join(xs_prepend, f"dsdxdy_{sign}_CC_iso.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}_CC_iso.fits")
            ))
            nc_paths.append((
                os.path.join(xs_prepend, f"dsdxdy_{sign}_NC_iso.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}_NC_iso.fits")
            ))
        elif is_new:
            cc_paths.extend([
                (os.path.join(xs_prepend, f"dsdxidy_{sign}-N-cc-charm-{oxygen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_{sign}-N-cc-charm-{oxygen_pdf}.fits")),
                (os.path.join(xs_prepend, f"dsdxidy_{sign}-N-cc-charm-{hydrogen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_{sign}-N-cc-charm-{hydrogen_pdf}.fits"))
            ])
            nc_paths.extend([
                (os.path.join(xs_prepend, f"dsdxidy_{sign}-N-nc-charm-{oxygen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_{sign}-N-nc-charm-{oxygen_pdf}.fits")),
                (os.path.join(xs_prepend, f"dsdxidy_{sign}-N-nc-charm-{hydrogen_pdf}.fits"),
                 os.path.join(xs_prepend, f"sigma_{sign}-N-nc-charm-{hydrogen_pdf}.fits"))
            ])
        else:
            cc_paths.append((
                os.path.join(xs_prepend, f"dsdxdy_{sign}-N-cc-charm-{hydrogen_pdf}.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}-N-cc-charm-{hydrogen_pdf}.fits")
            ))
            nc_paths.append((
                os.path.join(xs_prepend, f"dsdxdy_{sign}-N-nc-charm-{hydrogen_pdf}.fits"),
                os.path.join(xs_prepend, f"sigma_{sign}-N-nc-charm-{hydrogen_pdf}.fits")
            ))
        return {'cc': cc_paths, 'nc': nc_paths}

    else:
        raise ValueError(f"Unknown mode '{mode}'")


# ---------------------------------------------------------------------------
# Primary interaction lists
# ---------------------------------------------------------------------------

def build_standard_dis_interactions(config, primary_type):
    """Return a flat list of DISFromSpline objects (CC + NC as requested)."""
    xs_prepend   = get_xs_prepend(config)
    paths        = get_cross_section_paths(config, xs_prepend, mode="all")
    isoscalar_m  = float(0.938272 + 0.939565) / 2
    current_type = config.get("current_type", "both").lower()
    target_type  = siren.dataclasses.ParticleType.Nucleon
    xs_list      = []
    print(f"[INFO] Standard DIS XS paths:\n  CC: {paths['cc']}\n  NC: {paths['nc']}")

    if current_type in ("cc", "both"):
        for dsdy, sigma in paths['cc']:
            xs_list.append(
                siren.interactions.DISFromSpline(
                    dsdy, sigma, 1, isoscalar_m, 0.01,
                    [primary_type], [target_type], "m"
                )
            )
    if current_type in ("nc", "both"):
        for dsdy, sigma in paths['nc']:
            xs_list.append(
                siren.interactions.DISFromSpline(
                    dsdy, sigma, 2, isoscalar_m, 1.0,
                    [primary_type], [target_type], "m"
                )
            )
    return xs_list


def build_charm_interactions(config, primary_type):
    """
    Return (primary_xs_list, targets_list) for charm production.
    targets_list[i] corresponds to primary_xs_list[i].
    """
    xs_prepend   = get_xs_prepend(config)
    paths        = get_cross_section_paths(config, xs_prepend, mode="charm")
    isoscalar_m  = float(0.938272 + 0.939565) / 2
    current_type = config["current_type"].lower()
    print(f"[INFO] Charm XS paths:\n  CC: {paths['cc']}\n  NC: {paths['nc']}")

    if len(paths['cc']) == 2:
        targets = [
            siren.dataclasses.ParticleType.O16Nucleus,
            siren.dataclasses.ParticleType.HNucleus,
        ]
    else:
        targets = [siren.dataclasses.ParticleType.HNucleus]

    xs_list = []
    if current_type in ("cc", "both"):
        for idx, (dsdy, sigma) in enumerate(paths['cc']):
            xs_list.append(
                siren.interactions.QuarkDISFromSpline(
                    dsdy, sigma, 1, isoscalar_m, 1,
                    [primary_type], [targets[idx]], "m"
                )
            )
    if current_type in ("nc", "both"):
        for idx, (dsdy, sigma) in enumerate(paths['nc']):
            xs_list.append(
                siren.interactions.QuarkDISFromSpline(
                    dsdy, sigma, 2, isoscalar_m, 1,
                    [primary_type], [targets[idx]], "m"
                )
            )
    return xs_list


# ---------------------------------------------------------------------------
# Distribution builder
# ---------------------------------------------------------------------------

def build_injection_distributions(config):
    """Return (inj_dists_list, phys_dists_list) for Injector property setters."""
    e_min  = float(config["emin_gev"])
    e_max  = float(config["emax_gev"])
    gamma  = float(config.get("gamma", 2.0))
    edist  = siren.distributions.PowerLaw(gamma, e_min, e_max)
    mass_d = siren.distributions.PrimaryMass(0)

    if (config.get("fixed_dir_coszenith") is not None and
            config.get("fixed_dir_azimuth") is not None):
        print("[INFO] Using fixed direction injection.")
        coszen    = config["fixed_dir_coszenith"]
        phi       = config["fixed_dir_azimuth"]
        sinzen    = np.sqrt(max(0.0, 1.0 - coszen**2))
        vec       = siren.math.Vector3D(sinzen * np.cos(phi), sinzen * np.sin(phi), coszen)
        direction = siren.distributions.FixedDirection(vec)
        muon_rng  = siren.distributions.LeptonDepthFunction()
        position  = siren.distributions.ColumnDepthPositionDistribution(10, 10, muon_rng)
    else:
        print("[INFO] Using isotropic direction injection.")
        direction = siren.distributions.IsotropicDirection()
        muon_rng  = siren.distributions.LeptonDepthFunction()
        position  = siren.distributions.ColumnDepthPositionDistribution(600, 600.0, muon_rng)

    inj_dists  = [mass_d, edist, direction, position]
    phys_dists = [mass_d, edist, direction]
    return inj_dists, phys_dists


# ---------------------------------------------------------------------------
# Main simulation runner — new Injector/Weighter idiom
# ---------------------------------------------------------------------------

def run_simulation(config: dict, output: str, seed: int) -> str:
    """
    Run SIREN simulation using siren.injection.Injector (PR #71 API).

    Returns the output path prefix (without .parquet extension).

    All pybind11 objects are kept in local variables that outlive the
    GenerateEvents / SaveEvents calls to prevent dangling-pointer segfaults
    from premature GC.
    """
    events_to_inject = int(float(config["nev"]))
    nu_flavor        = config["flavor"]
    interaction      = config["interaction"].lower()

    if nu_flavor not in SIREN_PARTICLES:
        raise ValueError(f"Flavor code {nu_flavor} is not valid.")

    primary_type = SIREN_PARTICLES[nu_flavor]
    print(f"[INFO] Primary particle type: {primary_type.name}")

    # Detector model — always static orca18 loader.
    # The installed name in siren/resources/detectors/ is "orca18" (same string
    # the old SIREN_Controller("orca18", ...) accepted). "KM3NeTORCA" is the
    # YAML experiment label but is NOT a valid load_detector() key.
    print("[INFO] Loading detector model: orca18")
    detector_model = siren.utilities.load_detector("orca18")

    # ----- Build primary interactions -----
    if interaction == "charm":
        print("Configuring for CHARM production.")
        primary_xs_list = build_charm_interactions(config, primary_type)
    elif interaction == "all":
        print("Configuring for STANDARD DIS.")
        primary_xs_list = build_standard_dis_interactions(config, primary_type)
    else:
        raise ValueError(f"Invalid interaction type: {interaction}")

    # ----- Build distributions -----
    inj_dists, phys_dists = build_injection_distributions(config)

    # ----- Secondary interactions (charm only) -----
    # dict {ParticleType: [xs_objects]}  as required by Injector.secondary_interactions
    secondary_interactions             = {}
    secondary_injection_distributions  = {}
    secondary_physical_distributions   = {}

    if interaction == "charm":
        PT           = siren.dataclasses.ParticleType
        force_muonic = config.get("decay", "") == "muonic"
        if force_muonic:
            print("[INFO] Forcing D-meson decay into muonic channel (D -> K mu nu)")

        # Determine which D-meson types this primary flavor can produce.
        # QuarkDISFromSpline uses DTypesForPrimary() internally:
        #   nu  (positive PDG) -> DPlus, D0
        #   nubar (negative PDG) -> DMinus, D0Bar
        _is_nubar = primary_type in (
            PT.NuMuBar, PT.NuEBar, PT.NuTauBar)
        if _is_nubar:
            d_types = [PT.DMinus, PT.D0Bar]
        else:
            d_types = [PT.DPlus, PT.D0]

        d_eloss = siren.interactions.DMesonELoss()
        sec_vtx = siren.distributions.SecondaryPhysicalVertexDistribution()

        for d_type in d_types:
            d_decay = siren.interactions.CharmMesonDecay(
                primary_type=d_type, force_muonic=force_muonic
            )
            secondary_interactions[d_type]            = [d_eloss, d_decay]
            secondary_injection_distributions[d_type] = [sec_vtx]
            secondary_physical_distributions[d_type]  = []

        print("Charm secondaries wired:", [t.name for t in d_types])

    # ----- Injector -----
    injector = siren.injection.Injector()
    injector.seed                              = seed
    injector.number_of_events                  = events_to_inject
    injector.detector_model                    = detector_model
    injector.primary_type                      = primary_type
    injector.primary_interactions              = primary_xs_list
    injector.primary_injection_distributions   = inj_dists
    injector.stopping_condition                = lambda datum, i: False

    if secondary_interactions:
        injector.secondary_interactions            = secondary_interactions
        injector.secondary_injection_distributions = secondary_injection_distributions

    events, gen_times = GenerateEvents(injector)
    print(f"[INFO] Generated {len(events)} events")

    # ----- Weighter -----
    weighter = siren.injection.Weighter()
    weighter.injectors                       = [injector]
    weighter.detector_model                  = detector_model
    weighter.primary_type                    = primary_type
    weighter.primary_interactions            = primary_xs_list
    weighter.primary_physical_distributions  = phys_dists

    if secondary_interactions:
        weighter.secondary_interactions          = secondary_interactions
        weighter.secondary_physical_distributions = secondary_physical_distributions

    # ----- Save (.parquet + .hdf5 + .siren_events) -----
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    complete_path = os.path.abspath(output)
    if complete_path.endswith(".parquet"):
        complete_path = complete_path[:-8]

    print(f"[INFO] Saving events to: {complete_path}")
    SaveEvents(events, weighter, gen_times, fid_vol=None, output_filename=complete_path)

    return complete_path


# ---------------------------------------------------------------------------
# Particle / Interaction helpers for ROOT conversion (unchanged from legacy)
# ---------------------------------------------------------------------------

class Particle:
    def __init__(self, pdg, momentum, position, time, mother_id=None, status=None, energy=None, name=''):
        self.pdg       = pdg
        self.momentum  = momentum
        self.position  = position
        self.time      = time
        self.mother_id = mother_id
        self.status    = status
        self.energy    = energy
        self.name      = name

    def __repr__(self):
        return f"Particle({self.pdg}, {self.momentum}, {self.position}, {self.time})"


class Interaction:
    def __init__(self, parent, daughters, intermediates=None, event_id=None, meta=None):
        self.parent        = parent
        self.daughters     = list(daughters)
        self.intermediates = intermediates or []
        self.meta          = meta or {}

    def final_state_particles(self):
        return self.daughters

    def __repr__(self):
        return f"Interaction(parent={self.parent}, daughters={self.daughters})"


def create_interaction(event, time, decay_mode=""):
    """
    Build an Interaction from a single parquet row.

    The new SaveEvents schema is nested: primary_type / primary_momentum /
    vertex / secondary_types / secondary_momenta are per-interaction lists
    (one element per interaction in the event tree).  We pick index [0] for
    the primary interaction, exactly as the legacy code did — the charm-decay
    secondary interaction is index [1] when present.
    """
    # Primary neutrino (first interaction, depth 0)
    parent = Particle(
        pdg       = event["primary_type"][0],
        momentum  = event["primary_momentum"][0][1:],
        position  = event["vertex"][0],
        time      = time,
        mother_id = -1,
        status    = 0,
        energy    = event["primary_momentum"][0][0],
        name      = "primary"
    )

    daughters     = []
    intermediates = []

    primary_types           = event["primary_type"].to_list()
    secondary_types_nested  = event["secondary_types"].to_list()
    secondary_momenta_nested = event["secondary_momenta"].to_list()

    # Flatten across all interaction nodes in the tree
    flattened_types   = [item for sublist in secondary_types_nested  for item in sublist]
    flattened_momenta = [item for sublist in secondary_momenta_nested for item in sublist]

    for pdg, mom in zip(flattened_types, flattened_momenta):
        if pdg in primary_types:
            intermediates.append(Particle(
                pdg=pdg, momentum=mom[1:], position=event["vertex"][0],
                time=time, mother_id=1, status=2, energy=mom[0], name="intermediate"
            ))
        else:
            if abs(int(pdg)) > 1000000000:  # replace nuclear fragments with pion
                pdg = 211
            daughters.append(Particle(
                pdg=pdg, momentum=mom[1:], position=event["vertex"][0],
                time=time, mother_id=1, status=1, energy=mom[0], name="daughter"
            ))

    event_weight = float(event["event_weight"])

    # When forcing muonic decay, apply branching ratio correction.
    if decay_mode == "muonic":
        d_meson_pdg = next((p for p in primary_types if abs(p) in (411, 421)), None)
        if d_meson_pdg is not None and d_meson_pdg in BR_MUONIC:
            event_weight *= BR_MUONIC[d_meson_pdg]

    return Interaction(
        parent=parent, daughters=daughters,
        intermediates=intermediates,
        meta={"event_weight": event_weight}
    )


# ---------------------------------------------------------------------------
# Time / ROOT helpers (unchanged from legacy)
# ---------------------------------------------------------------------------

def CalcMJDFromTimeStamp(TimeStampSec, TimeStampTick):
    MJD  = TimeStampSec + TimeStampTick / 62500000
    MJD /= 24. * 3600.
    MJD += MJD0
    return MJD


def CalcTimeStamp(MJD):
    MJD0_local     = 40587
    PartFrac, PartInt = modf(((MJD - MJD0_local) * 24. * 3600.))
    PartFrac      *= 62500000
    return int(PartInt), int(PartFrac)


def interaction_to_root_event(interaction: Interaction, **kwargs):
    import ROOT  # deferred: needs aanet module loaded
    idx             = kwargs["idx"]
    options         = kwargs["options"]
    nevts           = options.n
    ene_factor      = kwargs["ene_factor"]

    TimeStampStart  = int(float(options.t.split(",")[0]))
    TimeStampStop   = int(float(options.t.split(",")[1]))
    can1, can2, can3 = (float(options.c.split(",")[0]),
                        float(options.c.split(",")[1]),
                        float(options.c.split(",")[2]))
    origin = ROOT.Vec(float(options.x.split(",")[0]),
                      float(options.x.split(",")[1]),
                      (can2 - can1) / 2)
    alpha  = options.a
    RNG    = np.random.default_rng(seed=options.r + idx)

    Revt             = ROOT.Evt()
    Revt.mc_run_id   = options.r
    Revt.mc_id       = idx
    Revt.id          = idx

    if "sorted_mjd" in kwargs:
        fMJD = kwargs["sorted_mjd"]
    else:
        MJDStart = CalcMJDFromTimeStamp(TimeStampStart, 0)
        MJDStop  = CalcMJDFromTimeStamp(TimeStampStop,  0)
        fMJD     = RNG.uniform(MJDStart, MJDStop)

    TimeStampSec, TimeStampTick = CalcTimeStamp(fMJD)
    Revt.mc_t = float(TimeStampSec - TimeStampStart) * 1e9 + float(TimeStampTick) * 16.0
    Revt.mc_event_time.SetSec(TimeStampSec)
    Revt.mc_event_time.SetNanoSec(TimeStampTick * 16)

    parent = interaction.parent
    pos1   = ROOT.Vec(parent.position[0], parent.position[1], parent.position[2])
    pos1  += origin
    trk    = ROOT.Trk()
    trk.id        = 0
    trk.mother_id = parent.mother_id
    trk.pos       = pos1
    pmag          = np.linalg.norm(parent.momentum)
    trk.dir       = ROOT.Vec(parent.momentum[0] / pmag,
                             parent.momentum[1] / pmag,
                             parent.momentum[2] / pmag)
    trk.E         = parent.energy
    trk.type      = int(parent.pdg)
    trk.t         = 0
    trk.status    = 100
    Revt.mc_trks.push_back(trk)

    Faanet = kwargs.get("flux_atm") or ROOT.Flux_Atmospheric()
    for i, daughter in enumerate(interaction.daughters):
        trk           = ROOT.Trk()
        trk.id        = i + 1
        trk.mother_id = daughter.mother_id
        pos2          = ROOT.Vec(daughter.position[0], daughter.position[1], daughter.position[2])
        pos2         += origin
        trk.pos       = pos2
        dmag          = np.linalg.norm(daughter.momentum)
        trk.dir       = ROOT.Vec(daughter.momentum[0] / dmag,
                                 daughter.momentum[1] / dmag,
                                 daughter.momentum[2] / dmag)
        trk.E         = daughter.energy
        trk.type      = int(daughter.pdg)
        trk.t         = 0
        trk.status    = 1
        Revt.mc_trks.push_back(trk)

    oneweight = (interaction.meta["event_weight"] * nevts * ene_factor
                 * ROOT.TMath.Power(Revt.mc_trks[0].E, alpha)
                 * 4 * ROOT.TMath.Pi())
    faanet = Faanet.dNdEdOmega(Revt.mc_trks[0].type,
                                ROOT.TMath.Log10(Revt.mc_trks[0].E),
                                Revt.mc_trks[0].dir.z)
    Revt.w = [interaction.meta["event_weight"], oneweight, faanet * oneweight]
    return Revt


# ---------------------------------------------------------------------------
# gSeaGen ROOT file writer (unchanged from legacy)
# ---------------------------------------------------------------------------

def write_gSeaGen_rootfile(siren_filename, options, decay_mode=""):
    import ROOT        # deferred: needs aanet module loaded
    from tqdm import tqdm  # deferred: optional dependency
    import aa          # noqa: F401 — needed to initialise ROOT event classes via side-effect
    if not siren_filename.endswith(".parquet"):
        raise ValueError("Input file is not a parquet file")

    data  = ak.from_parquet(siren_filename)
    nevts = len(data)

    if nevts != options.n:
        print(f"Warning: Expected {options.n} events, parquet has {nevts}")

    TimeStampStart = int(float(options.t.split(",")[0]))
    TimeStampStop  = int(float(options.t.split(",")[1]))

    complete_path = siren_filename[:-8] + ".root"

    can1, can2, can3 = (float(options.c.split(",")[0]),
                        float(options.c.split(",")[1]),
                        float(options.c.split(",")[2]))
    outfile = ROOT.OutputEventFile(complete_path)
    header  = ROOT.Head()
    header.set_line("start_run",     str(options.r))
    header.set_line("seed",          str(options.r))
    header.set_line("cut_ene",       str(options.e) + " " + str(options.me))
    header.set_line("alpha",         str(options.a))
    header.set_line("nupdg",         str(options.f))
    header.set_line("model",         str(options.M))
    header.set_line("fixedcan",      str(options.x.replace(",", " ")) + " " + str(options.c.replace(",", " ")))
    header.set_line("depth",         str(options.d))
    header.set_line("time_interval", str(TimeStampStart) + " " + str(TimeStampStop))
    header.set_line("simul",         "gSeaGen")
    intvol = ROOT.TMath.Pi() * can3 * can3 * (can2 - can1) * 1.03975
    header.set_line("genvol",        str(can1) + " " + str(can2) + " " + str(can3) + " " + str(intvol) + " " + str(nevts))
    print("Header:\n", header)
    outfile.write_header(header)

    alpha      = options.a
    ene_factor = (ROOT.TMath.Power(options.me, -alpha + 1) -
                  ROOT.TMath.Power(options.e,  -alpha + 1)) / (-alpha + 1)

    MJDStart   = CalcMJDFromTimeStamp(TimeStampStart, 0)
    MJDStop    = CalcMJDFromTimeStamp(TimeStampStop,  0)
    RNG_time   = np.random.default_rng(seed=options.r)
    sorted_MJDs = np.sort(RNG_time.uniform(MJDStart, MJDStop, size=nevts))

    flux_atm = ROOT.Flux_Atmospheric()  # instantiate once — daemonflux tables loaded here
    for idx, evt in tqdm(enumerate(data), total=len(data)):
        interaction = create_interaction(evt, time=0, decay_mode=decay_mode)
        R_evt = interaction_to_root_event(
            interaction, options=options, idx=idx,
            ene_factor=ene_factor, sorted_mjd=sorted_MJDs[idx], flux_atm=flux_atm
        )
        outfile.write(R_evt)

    del outfile
    sys.exit(0)


# ---------------------------------------------------------------------------
# CAN reader (unchanged from legacy)
# ---------------------------------------------------------------------------

def read_can_txt(filename):
    variables = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split("=", 1)
            key   = key.strip()
            value = value.strip()
            try:
                value = float(value)
            except ValueError:
                pass
            variables[key] = value
    return variables


# ---------------------------------------------------------------------------
# Deprecated stub — kept so old call sites don't crash
# ---------------------------------------------------------------------------

def CreateTempSirenDetector(detname, height, radius, depth):
    """Deprecated — kept for backward compatibility. Returns 'KM3NeTORCA'."""
    return "KM3NeTORCA"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
        description="Run KM3NeT SIREN Simulation — modern Injector/Weighter API.")
    parser.add_argument('--config',        required=True, type=json.loads,
                        help="Settings dictionary as JSON string.")
    parser.add_argument('--output',        type=str, default="output",
                        help="Output path prefix (without .parquet extension)")
    parser.add_argument('--seed',          type=int, default=1, help="Random seed")
    parser.add_argument('--detector_info', type=str,
                        help="Detector CAN info .txt file (from JPrintDetector)")

    args = parser.parse_args()

    # Read detector CAN parameters
    can_info = read_can_txt(args.detector_info)
    can1     = float(can_info['CAN_Zmin_M'])
    can2     = float(can_info['CAN_Zmax_M'])
    can3     = float(can_info['CAN_Radius_M'])

    print(f"[INFO] Using static detector: orca18")
    path = run_simulation(
        config=args.config,
        output=args.output,
        seed=args.seed
    )

    # ROOT conversion is handled by parquet_to_root_km3net.py (Step 2 in siren.smk).
    print(f"[INFO] Injection complete — parquet: {path}.parquet")


if __name__ == "__main__":
    main()
