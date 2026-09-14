# Upstream

This service is Model Loader by scratchhax, folded into noevia:
https://github.com/scratchhax/model-loader at commit
`e11a6ec307fa405144678930d507046163369b46`, MIT License (see `LICENSE`).

The first commit that added this directory contains the upstream files unchanged, apart
from omitting the documentation screenshots. Every later change is a noevia change and is
visible in git history. noevia drives it through the JSON API in `app/api.py` and renders
all screens in its own web app; the original server-rendered pages remain only until the
native screens replace them.
