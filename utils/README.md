# Utils folder

The current tested version of `app.py` already defines the helper functions it uses, including dataset/model loading, target-label normalization, feature-column selection, and prediction-form construction.

To avoid changing working prediction code close to the submission deadline, these functions have **not** been copied or moved into this folder. The `__init__.py` file simply makes `utils` a valid Python package for future development. The app does not need to import anything from `utils` in the current version.

Do not duplicate the existing helpers here unless you also update `app.py` to import the shared implementation and retest all five modules.
