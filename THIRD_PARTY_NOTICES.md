# Third-party notices

## CPython

Windows distributions include an unmodified CPython embeddable package under
the Python Software Foundation License Version 2. The archive includes
Python's license notice. Provenance and its official SHA-256 are recorded in
`vendor/python/python-runtime.json`.

Windows distributions also include PyPA's unmodified `get-pip.py` bootstrap
script. Its provenance and SHA-256 are recorded in the same manifest; pip's
license metadata is retained by its bundled installation payload.

## Socket Firewall Free

Windows distributions include an unmodified Socket Firewall Free executable.
Socket Firewall Free is provided by Socket Security, Inc. under the PolyForm
Shield License 1.0.0. Recipients can read the complete terms at:

https://github.com/SocketDev/sfw-free/blob/main/README.md#license

Pinned release provenance and SHA-256 values are recorded in
`vendor/security-tools.json`.

## uv

Linux and macOS preparation downloads a pinned, unmodified `uv` release from
Astral. `uv` is available under the Apache License 2.0 or MIT License. Release
provenance and per-platform archive SHA-256 values are recorded in
`vendor/security-tools.json`.
