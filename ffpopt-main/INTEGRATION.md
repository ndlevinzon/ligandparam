# Upstream ffpopt tree

The **runtime Python package** used by ligandparam now lives at
[`src/ffpopt`](../src/ffpopt).

This directory keeps the original ffpopt project docs, examples, CMake/RESP
build, and environment files. Prefer importing ``ffpopt`` from ``src/ffpopt``
(installed via the ligandparam ``pyproject.toml``) rather than installing this
tree separately unless you need the full upstream build.
