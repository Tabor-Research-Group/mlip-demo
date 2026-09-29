"""Run an ASE optimizer while leaving the choice of method visible to callers."""

from ase.optimize import BFGS


def optimize_ase(atoms_or_filter, *, fmax, steps=None, optimizer=BFGS,
                 callback=None, interval=1, **optimizer_kwargs):
    """Optimize an ASE Atoms object or cell filter in place and return the optimizer.

    The caller attaches a calculator and chooses any constraints before this call.
    Pass an ASE optimizer class such as ``BFGS`` or ``FIRE``. Returning the
    optimizer exposes ``nsteps`` and ``converged()`` for notebook analysis.
    ``callback`` can record a trajectory at the requested ``interval``.
    Remaining keyword arguments go to the optimizer constructor, for example
    ``logfile=None``.
    """
    dynamics = optimizer(atoms_or_filter, **optimizer_kwargs)
    if callback is not None:
        dynamics.attach(callback, interval=interval)
    run_kwargs = {"fmax": fmax}
    if steps is not None:
        run_kwargs["steps"] = steps
    dynamics.run(**run_kwargs)
    return dynamics
