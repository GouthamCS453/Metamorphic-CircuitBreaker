import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.metamorphic_families import SEVERITY_WEIGHTS, HIERARCHICAL_TEST_MATRIX
from src.circuit_breaker import CircuitBreakerConfig

fails = 0


def check(ok, msg):
    global fails
    print(("[OK]   " if ok else "[FAIL] ") + msg)
    fails += int(not ok)


# 1. Severity weights must descend
w = SEVERITY_WEIGHTS
check(
    w["mild"] > w["moderate"] > w["severe"],
    f"severity weights descending: {w}"
)


# 2. Exactly 3 transform types per family
fam = {}

for t in HIERARCHICAL_TEST_MATRIX:
    fam.setdefault(t.family, set()).add(t.transform_type)

sizes = {k: len(v) for k, v in fam.items()}

check(
    len(set(sizes.values())) == 1 and set(sizes.values()) == {3},
    f"M_k == 3 for every family: {sizes}"
)


# 3. No identity transforms
noop = [
    t
    for t in HIERARCHICAL_TEST_MATRIX
    if getattr(t, "is_identity", False)
    or str(getattr(t, "severity_label", "")).lower()
    in ("none", "identity")
]

check(
    not noop,
    f"no identity tiers (found {len(noop)})"
)


# 4. theta_warn must always be below theta_trip
ok = True

for trip in [0.05, 0.10, 0.20, 0.30, 0.55, 0.80]:
    c = CircuitBreakerConfig(theta_trip=trip)
    ok &= c.theta_warn < c.theta_trip

check(
    ok,
    "theta_warn < theta_trip for every theta_trip"
)


# 5. Fixed CBI weights
c = CircuitBreakerConfig()

check(
    (c.alpha, c.beta, c.gamma) == (0.50, 0.35, 0.15),
    f"alpha/beta/gamma = {(c.alpha, c.beta, c.gamma)}"
)


print(f"\n{fails} failing checks")