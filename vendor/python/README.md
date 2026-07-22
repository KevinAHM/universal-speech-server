# Bundled Windows Python payload

This directory contains the Windows runtime payload that `start_server.bat`
installs: the unmodified CPython embeddable archive named by
`python-runtime.json`, the pinned `get-pip.py`, and their generated
`bundle-manifest.json`. End-user startup never downloads Python from
python.org. The recorded upstream URLs are provenance and every shipped file
must match its pinned SHA-256.

The release bundle also contains the pinned `get-pip.py` recorded in the
manifest. The bootstrapper uses it only when the embedded interpreter has no
working pip, then runs the dependency installation as
`bin/sfw.exe python.exe -m pip install -r requirements-runtime.txt`. It never
installs into a system interpreter.

`scripts/build_windows_runtime.py` copies these payloads into a release-staging
overlay and verifies both pinned Python bootstrap files. That builder also
ships the pinned `bin/sfw.exe` recorded in `vendor/security-tools.json`.
