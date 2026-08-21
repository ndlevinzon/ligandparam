"""Compat re-export; canonical: :mod:`ffpopt.geom.geometric`.

Also the ``python -m ffpopt.geometric_compat`` entry used by geomopt subprocess.
"""

from ffpopt.geom.geometric import *  # noqa: F401,F403
from ffpopt.geom.geometric import main

if __name__ == "__main__":
    main()
