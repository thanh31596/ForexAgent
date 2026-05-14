"""FastAPI service layer.

Modules
-------
main      FastAPI application factory, middleware (latency, logging,
          exception handlers), lifespan events.
routes    Endpoint definitions: ``/predict``, ``/health``, ``/metrics``,
          ``/retrain``. OpenAPI docs auto-generated at ``/docs``.
"""
