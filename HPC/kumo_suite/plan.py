"""Predeclared one-factor experiments; no validation-driven feature selection."""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Experiment:
    name: str
    size: str = "large"
    view: str = "compact"
    context: int = 4096
    bags: int = 1
    estimators: int = 4
    segment: str = "global"
    sampling: str = "stratified"
    transform: str = "native"
    drop: str = "none"
    seed: int = 42


def experiments():
    base = Experiment("large_base")
    out = [replace(base, name=f"{size}_base", size=size) for size in ("small", "medium", "large")]
    for size in ("small", "medium", "large"):
        for view in ("enhanced", "wide"):
            out.append(replace(base, name=f"{size}_{view}", size=size, view=view))
    for n in (1024, 2048, 8192, 16384, 32768):
        out.append(replace(base, name=f"large_ctx{n}", context=n))
    for n in (1, 2, 8, 16):
        out.append(replace(base, name=f"large_est{n}", estimators=n))
    for n in (2, 4, 8):
        out.append(replace(base, name=f"large_bag{n}", bags=n))
    for segment in ("population", "soft"):
        for view in ("compact", "enhanced"):
            out.append(replace(base, name=f"large_{segment}_{view}", segment=segment, view=view))
    for sampling in ("uniform", "coverage"):
        for bags in (1, 4):
            out.append(replace(base, name=f"large_{sampling}_bag{bags}", sampling=sampling, bags=bags))
    for transform in ("log", "rank", "median_indicator"):
        for view in ("compact", "enhanced"):
            out.append(replace(base, name=f"large_{view}_{transform}", transform=transform, view=view))
    for drop in ("repay", "ecom", "bureau", "acct", "prev", "pricing", "calendar"):
        out.append(replace(base, name=f"large_drop_{drop}", view="enhanced", drop=drop))
    # Confirmation is explicit rather than multiplying the entire expensive grid.
    for seed in (2026, 999):
        for size in ("small", "medium", "large"):
            out.append(replace(base, name=f"{size}_seed{seed}", size=size, seed=seed))
        for segment in ("population", "soft"):
            out.append(replace(base, name=f"large_{segment}_seed{seed}", segment=segment, seed=seed))
    return {x.name: x for x in out}


def select(preset="screen", names=None):
    all_exp = experiments()
    if names:
        return [all_exp[n] for n in names]
    if preset == "screen":
        return [all_exp[n] for n in ("small_base", "medium_base", "large_base", "large_enhanced",
                                     "large_population_compact", "large_soft_compact")]
    if preset == "confirm":
        return [x for x in all_exp.values() if x.seed != 42]
    if preset == "all":
        return list(all_exp.values())
    raise ValueError(preset)


def records(preset, names=None):
    return [asdict(x) for x in select(preset, names)]
