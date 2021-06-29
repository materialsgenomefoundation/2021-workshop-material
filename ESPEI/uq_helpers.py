import numpy as np
from pycalphad import variables as v
from espei.utils import database_symbols_to_fit, formatted_parameter

def _highest_density_indices(flat_lnprob, credible_interval=0.95):
    """Return the indices for the for the points with the highest density
    
    Parameters
    ----------
    flat_lnprob : ArrayLike[np.float_]
        1D array of probabilities
    credible_interval : float
        Fraction on [0, 1] of the highest density points to consider. A credible_interval of 0.95 means the indices corresponding to the 95% HDI.
        
    Returns
    -------
    ArrayLike[np.int_]
        1D array of indices in the credible interval
    """
    cutoff_probability = np.quantile(flat_lnprob, 1-credible_interval, interpolation='lower')
    return np.nonzero(flat_lnprob > cutoff_probability)[0]


def highest_density_parameters(trace, lnprob, credible_interval=0.95, burn_in=0, thin=1):
    """Return the parameter trace with the highest density according to the credible_interval.
    
    The trace will be flattened to 2D.
    
    Parameters
    ----------
    trace : ArrayLike[np.float_]
        3D array of shape (chains, iterations, parameters)
    lnprob : ArrayLike[np.float_]
        2D array of shape (chains, iterations)
    credible_interval : float
        Fraction on [0, 1] of the highest density points to consider. A credible_interval of 0.95 means the indices corresponding to the 95% HDI.
    burn_in : int
        Number of iterations to consider as burn in
    thin : int
        Take every n-th sample
    
    Returns
    -------
    ArrayLike[np.float_]
        2D array of shape (samples, parameters)
    
    """
    flat_lnprob = lnprob[:, burn_in:].reshape(-1)
    hdi_idx = _highest_density_indices(flat_lnprob, credible_interval)
    flat_trace = trace[:, burn_in:, :].reshape(-1, trace.shape[-1])
    return flat_trace[hdi_idx, :][::thin, :]

from sympy import S, log, Symbol
from dataclasses import dataclass
@dataclass
class FormattedParameter():
    phase_name: str
    interaction: str
    symbol: Symbol
    term: Symbol
    parameter_type: str
    term_symbol: Symbol
    
    def _repr_latex_(self):
        COEFFICIENT_MAP = {
            S.One: 'a',
            v.T: 'bT',
            v.T*log(v.T): 'cT\\lnT',
        }
        escaped_phase_name = self.phase_name.replace('_', '\_')  # escape underscores
        coeff = COEFFICIENT_MAP[self.term]
        if self.parameter_type.startswith('L'):
            order = self.parameter_type[1:]
            return f'${{}}^{order} L^\\mathrm{{{escaped_phase_name}}}_{{{self.interaction}}}: {coeff}$'
        return f'${self.parameter_type}^\\mathrm{{{escaped_phase_name}}}_\\mathrm{{{self.interaction}}}: {coeff}$'
    
    @classmethod
    def from_nt(cls, fp):  # named tuple FormattedParameter
        return cls(fp.phase_name, fp.interaction, fp.symbol, fp.term, fp.parameter_type, fp.term_symbol)

def format_parameter_symbols(dbf):
    return [FormattedParameter.from_nt(formatted_parameter(dbf, sym))._repr_latex_() for sym in database_symbols_to_fit(dbf)]
