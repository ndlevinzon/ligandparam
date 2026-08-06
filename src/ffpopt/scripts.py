import sys
import subprocess
import os
from importlib import resources

def _run_bin_script(script_name):
    """Internal helper to locate and run scripts via the current Python interp."""
    try:
        # Modern Python 3.9+ way to find files inside a package folder
        import ffpopt.bin as bin_pkg
        script_path = str(resources.files(bin_pkg).joinpath(script_name))
    except (ImportError, AttributeError):
        # Fallback for older environments
        import ffpopt
        script_path = os.path.join(os.path.dirname(ffpopt.__file__), "bin", script_name)

    if not os.path.exists(script_path):
        print(f"Error: {script_name} not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    # Execute using current python to bypass +x permission issues
    cmd = [sys.executable, script_path] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

# These simple wrappers detect their own name and call the helper
def ffpopt_DihedScan():           _run_bin_script("ffpopt-DihedScan.py")
def ffpopt_Optimize():            _run_bin_script("ffpopt-Optimize.py")
def ffpopt_GenDihedFit():         _run_bin_script("ffpopt-GenDihedFit.py")
def ffpopt_DihedTwistWorkflow():  _run_bin_script("ffpopt-DihedTwistWorkflow.py")
def ffpopt_DihedWavefront():      _run_bin_script("ffpopt-DihedWavefront.py")
def ffpopt_ConfSearch():          _run_bin_script("ffpopt-ConfSearch.py")
def ffpopt_RespFit():             _run_bin_script("ffpopt-RespFit.py")
def ffpopt_DeltaRespFit():        _run_bin_script("ffpopt-DeltaRespFit.py")
def ffpopt_CpeFit():              _run_bin_script("ffpopt-CpeFit.py")
def ffpopt_WavefrontToDP():       _run_bin_script("ffpopt-WavefrontToDP.py")
def ffpopt_WavefrontAnimate():    _run_bin_script("ffpopt-WavefrontAnimate.py")
def ffpopt_DihedTwistAnimate():   _run_bin_script("ffpopt-DihedTwistAnimate.py")
def ffpopt_xyz2mol2():            _run_bin_script("ffpopt-xyz2mol2.py")
def ffpopt_PrepareInput():        _run_bin_script("ffpopt-PrepareInput.py")
def ffpopt_Json2Crds():           _run_bin_script("ffpopt-Json2Crds.py")
def ffpopt_JsonJoin():            _run_bin_script("ffpopt-JsonJoin.py")
def ffpopt_JsonSplit():           _run_bin_script("ffpopt-JsonSplit.py")
def ffpopt_Json2Img():            _run_bin_script("ffpopt-Json2Img.py")
def ffpopt_FindSugarPuckers():    _run_bin_script("ffpopt-FindSugarPuckers.py")
def ffpopt_NDimWavefront():       _run_bin_script("ffpopt-NDimWavefront.py")
def ffpopt_DeltaPuckerFit():      _run_bin_script("ffpopt-DeltaPuckerFit.py")
