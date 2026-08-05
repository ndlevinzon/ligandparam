Example 03: Modifying Recipe Stages
===================================

This example starts from :class:`~ligandparam.recipes.LazyLigand`, then removes
and inserts stages before execution.

Learning outcomes
-----------------

1. Remove a stage by name with ``remove_stage``.
2. Insert a custom stage before another named stage with ``insert_stage``.

Files
-----

See ``examples/03_ModifySteps`` in the source tree.

Tutorial
--------

Create the recipe, call ``setup()``, and inspect the default stage list:

.. literalinclude :: ../../../../examples/03_ModifySteps/param_thio.py
    :language: python
    :end-before: test.remove_stage

Remove the first charge-normalization stage:

.. literalinclude :: ../../../../examples/03_ModifySteps/param_thio.py
    :language: python
    :start-at: test.remove_stage
    :end-at: test.remove_stage

Insert a replacement normalization stage before ``MinimizeLowTheory``:

.. literalinclude :: ../../../../examples/03_ModifySteps/param_thio.py
    :language: python
    :start-at: test.insert_stage
    :end-at: "MinimizeLowTheory"

Then list stages again and execute as usual:

.. literalinclude :: ../../../../examples/03_ModifySteps/param_thio.py
    :language: python
    :start-after: after inserting

Full code
---------

.. literalinclude :: ../../../../examples/03_ModifySteps/param_thio.py
    :language: python
