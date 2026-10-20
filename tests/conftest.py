import pytest

from datadiff_agent.contracts import load_contract
from datadiff_agent.demo import make_day
from datadiff_agent.memory import Memory
from datadiff_agent.tools import RunContext


@pytest.fixture
def contract():
    return load_contract("bookings")


@pytest.fixture
def memory():
    mem = Memory(":memory:")
    yield mem
    mem.close()


@pytest.fixture
def make_ctx(contract, memory):
    """Build a RunContext comparing demo day `new_day` against day 2."""

    def _make(new_day: int, old_day: int = 2) -> RunContext:
        return RunContext(
            dataset="bookings",
            batch=f"day{new_day}.csv",
            old=make_day(old_day),
            new=make_day(new_day),
            contract=contract,
            memory=memory,
        )

    return _make
