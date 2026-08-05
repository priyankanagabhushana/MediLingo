# Local data

The datasets are downloaded and prepared locally on Thor and are intentionally
not committed to the private GitHub repository. The repository keeps the
download metadata, checksums, preparation scripts, and evaluation reports that
make the data workflow reproducible.

Sources used by the project include:

- ahazeemi/opus-medical-en-de
- OPUS EMEA English-German data
- qanastek/EMEA-V3 only as a labelled fallback when necessary

Run scripts/download_data.py and scripts/prepare_data.py inside the project
environment to recreate the local data directories. Review each source's
license and usage terms before redistribution.
