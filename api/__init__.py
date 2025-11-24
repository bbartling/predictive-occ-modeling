"""
API package for predictive occupancy modeling.

The API is implemented using FastAPI and exposes three endpoints:
``/train`` to train a model from a CSV file, ``/state`` to query
the current occupancy and next transition, and ``/schedule`` to
retrieve the weekly schedule matrix.  The actual modeling logic
lives in the ``predictive_occ`` package.

To run the API with uvicorn:

    uvicorn api.app:app --host 0.0.0.0 --port 8000

"""

# intentionally empty; see ``api.app`` for the FastAPI application