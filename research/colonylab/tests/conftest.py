import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="session", autouse=True)
def bound_native_threads():
    # Tiny deterministic fixtures run faster without large BLAS thread pools.
    with threadpool_limits(limits=1):
        yield
