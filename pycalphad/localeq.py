from pycalphad.core.eqsolver import pointsolve
from pycalphad.core.solver import Solver
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.utils import unpack_components, instantiate_models
from pycalphad.codegen.callables import build_phase_records
from pycalphad import calculate, variables as v
import numpy as np

def starting_point(dbf, species, phase, conds, model):
    
    # Broadcasting conditions not supported
    cur_conds = {str(k): float(v) for k, v in conds.items()}
    # callables are being unnecessarily rebuilt here
    calc_p = calculate(dbf, species, phase, T=cur_conds.get('T', 300), P=cur_conds['P'],
                       pdens=10, model=model, callables=None)
    points_p = np.atleast_1d(np.ones(len(calc_p.points), dtype=bool))
    comp_tol = 0.1
    # Find configurations of desired composition
    for key, val in conds.items():
        if not isinstance(key, v.MoleFraction):
            continue
        comp = str(key)[2:]
        points_p &= (np.abs(calc_p.X.sel(component=comp) - val).values.squeeze() < comp_tol)
    if not np.any(points_p):
        # no points within tolerance, just pick the minimum energy point
        idx_p = np.argmin(calc_p.GM.values.squeeze())
    else:
        # choose minimum energy point within tolerance
        local_idx_p = np.argmin(np.atleast_1d(calc_p.GM.values.squeeze())[points_p])
        idx_p = np.arange(len(points_p))[points_p][local_idx_p]
    state_variables = np.array([cur_conds.get('N', 1.0), cur_conds.get('P', 1e5), cur_conds.get('T', 300)])
    site_fractions = np.array(calc_p.Y.isel(points=idx_p).values.squeeze())
    phase_amt = 1.0 # arbitrary
    return site_fractions, phase_amt, state_variables
    
def composition_set(dbf, comps, phase, conds, phase_amt=None, fixed=None, models=None, phase_records=None, parameters=None):
    species = sorted(unpack_components(dbf, comps), key=str)
    models = models if models is not None else {}
    phase_records = phase_records if phase_records is not None else {}
    if models.get(phase, None) is None:
        models[phase] = instantiate_models(dbf, species, [phase], parameters=parameters)[phase]
    if phase_records.get(phase, None) is None:
        phase_records[phase] = build_phase_records(dbf, species, [phase],
                                                   conds, models, build_gradients=True, build_hessians=True)[phase]
    compset = CompositionSet(phase_records[phase])
    if fixed is not None:
        compset.fixed = fixed
    site_fractions, _phase_amt, state_variables = starting_point(dbf, species, phase, conds, models[phase])
    if phase_amt is None:
        phase_amt = _phase_amt
    compset.update(site_fractions, phase_amt, state_variables)
    return compset

def local_equilibrium(composition_sets, comps, conds):
    # Broadcasting conditions not supported
    cur_conds = {str(k): float(v) for k, v in conds.items()}
    solver = Solver()
    comps = sorted([v.Species(x) for x in comps])
    result = pointsolve(composition_sets, comps, cur_conds, solver)
    return result, composition_sets