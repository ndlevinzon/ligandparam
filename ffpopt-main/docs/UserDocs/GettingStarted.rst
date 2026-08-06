Getting Started
================


Welcome to ffpopt, a tool developed by the York Group to generate bespoke force-field parameters for drug-like moleucles. This documentation will guide you through the initial steps to get started with ffpopt.


Installation
------------

1. Download the miniforge installer
   
   .. code-block:: bash

       wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

2. Install miniforge
   
   .. code-block:: bash

       bash ./Miniforge3-Linux-x86_64.sh -b -f -p ${PWD}/miniforge3

3. Enter the conda environment and create ffpopt environments for pytorch and tensorflow.  The `eval` command is to setup your shell to use conda environments if you have not setup conda on your computer before. If you've already used conda environments on your computer, then it is unnecessary. Simply activate your base environment and create ffpopt environment(s) using the environment.yml file.
   
   .. code-block:: bash
		   
       source ${PWD}/miniforge3/bin/activate
       mamba env create --yes -n ffpopt-pytorch -f environment.yml
       mamba env create --yes -n ffpopt-tensorflow -f environment.yml
       eval "$(mamba shell hook --shell bash)"

4. Install ffpopt source and python dependencies via pip. The dependencies are separated into "groups".

     a. -\-group=fairchem : installs https://github.com/facebookresearch/fairchem
     b. -\-group=tensorflow : installs https://github.com/deepmodeling/deepmd-kit and relevant dependencies
     c. -\-group=pytorch : installs pytorch libraries and pytorch models

    The 3 groups install incompatible software stacks.
    The fairchem models aren't very good, so one doesn't normally install
    the fairchem group.
    The tensorflow group will install deepmd-kit used to evaluate the
    qdpi2 model, which is distributed along with ffpopt.
    The pytorch group is a very popular framework to develop machine learning
    models.
    You will need to create separate environments for each stack.
    For example, -\-group tensorflow will install deepmd-kit[tf],
    which is incompatible with the -\-group pytorch which installs
    deepmd-kit[torch].

    If ACADEMIC=TRUE is specified, then machine learning
    models will be downloaded from the internet for academic
    use only. The default behavior is to assume the user is
    in industry, in which case you'll need to contact the
    appropriate authors for permission to use their trained
    models.

   .. code-block:: bash
		   
       mamba activate ffpopt-tensorflow
       ACADEMIC=TRUE python3 -m pip install --group tensorflow --extra-index-url https://download.pytorch.org/whl/cu121 .
       mamba deactivate

       mamba activate ffpopt-pytorch
       ACADEMIC=TRUE python3 -m pip install --group pytorch --extra-index-url https://download.pytorch.org/whl/cu121 .
       mamba deactivate

5. On modern OSs, you may need to remove the exec-flag on the shared libraries installed by conda/mamba.
   
   .. code-block:: bash
		   
       execstack -c ${PWD}/miniforge3/envs/ffpopt-tensorflow/lib/*.so
       execstack -c ${PWD}/miniforge3/envs/ffpopt-pytorch/lib/*.so
		   

6. Use the following commands to make the documentation in HTML format.
   
   .. code-block:: bash
		   
       python3 -m pip install sphinx sphinx_rtd_theme
       cd docs; make html; xdg-open _build/html/index.html

       
