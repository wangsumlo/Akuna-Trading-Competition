import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id


# ============================================================================
# YOUR MARKET MAKER -- fill in the six stubbed methods below
# ============================================================================


class MarketMaker:
    # --- strategy / risk knobs -------------------------------------------------
    RATE_STEP: float = 0.25
    RATE_TARGET: float = 2.0
    HALF_EDGE: float = 0.02
    CASH_FLOOR_FRACTION: float = 0.25
    QUOTE_RISK_FRACTION: float = 0.10
    FOK_RISK_FRACTION: float = 0.10
    MAX_QTY: int = 100000
    MAX_SKEW: float = 0.05
    FOK_MIN_EDGE: float = 0.01

    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:
        self.underlying_state: list[Underlying] = list(underlying_initial_state)
        self.active_option_state: list[BinaryOption] = list(option_initial_state)
        self.cash_balance: float = float(cash_balance)
        self.initial_cash: float = float(cash_balance)
        self.position: Position = Position()

        # Estimated parameters (filled in by warm_up).
        self._params: MarketParameters = self._default_params()

        # Risk-control derived values.
        self._cash_floor: float = self.CASH_FLOOR_FRACTION * self.initial_cash
        self._quote_max_loss: float = self.QUOTE_RISK_FRACTION * self.initial_cash
        self._fok_max_loss: float = self.FOK_RISK_FRACTION * self.initial_cash

        # Our own position accounting (long and short legs tracked separately so
        # settlement credits can be computed exactly).
        self._long_qty: dict[int, int] = defaultdict(int)
        self._short_qty: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------ basics
    @property
    def name(self) -> str:
        return "BuffyMM"

    def _default_params(self) -> MarketParameters:
        return MarketParameters(
            ajarai_drift=0.0,
            ajarai_idio_std_dev=0.05,
            ajarai_rate_beta=0.0,
            ajarai_sector_beta=0.5,
            rate_down_probability=0.30,
            rate_reversion_strength=0.10,
            rate_up_probability=0.30,
            sector_std_dev=0.05,
            theriodic_drift=0.0,
            theriodic_idio_std_dev=0.05,
            theriodic_rate_beta=0.0,
            theriodic_sector_beta=0.5,
            rate_step=self.RATE_STEP,
            rate_target=self.RATE_TARGET,
        )

    # ------------------------------------------------------------- state hooks
    def on_step_advance(
        self, new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]
    ) -> None:
        # Settle options that just expired. A held option with steps_until_expiry == 1
        # in the old state expires on this advance; an option missing from the new
        # state has been removed (also expiry). Valuation uses the new underlying
        # values, matching "payoffs credited at end of day".
        new_vals: dict[int, float] = {u.underlying_id: u.value for u in new_underlying_state}
        new_ids: set[int] = {o.option_id for o in new_option_state}
        old_options: dict[int, BinaryOption] = {o.option_id: o for o in self.active_option_state}

        held_ids: set[int] = set(self._long_qty.keys()) | set(self._short_qty.keys())
        for option_id in held_ids:
            old_opt: BinaryOption | None = old_options.get(option_id)
            if old_opt is None:
                continue
            if old_opt.steps_until_expiry <= 1 or option_id not in new_ids:
                valuation: float = old_opt.expiry_valuation(new_vals)
                long_qty: int = self._long_qty.get(option_id, 0)
                short_qty: int = self._short_qty.get(option_id, 0)
                # Long position credits qty * V; short position credits qty * (1 - V).
                self.cash_balance += long_qty * valuation + short_qty * (1.0 - valuation)
                self._long_qty.pop(option_id, None)
                self._short_qty.pop(option_id, None)
                self.position.option_quantity_by_option_id[option_id] = 0

        self.underlying_state = list(new_underlying_state)
        self.active_option_state = list(new_option_state)

    def on_trade(
        self, option: BinaryOption, price: float, quantity: int, counterparty_id: int
    ) -> None:
        # `quantity` is signed: positive = we bought, negative = we sold.
        self.position.add_option_quantity(option.option_id, quantity)
        if quantity > 0:
            # Buying reserves the premium (max loss if it expires worthless).
            self.cash_balance -= price * quantity
            self._long_qty[option.option_id] += quantity
        elif quantity < 0:
            # Selling reserves the worst-case payout (max loss if it expires ITM).
            qty: int = -quantity
            self.cash_balance -= (1.0 - price) * qty
            self._short_qty[option.option_id] += qty

    # ------------------------------------------------------------- pricing core
    def price_option(self, option: BinaryOption) -> float:
        return self._price_with_params(self._params, option)

    def price_option_from_parameters(
        self, market_parameters: MarketParameters, option: BinaryOption
    ) -> float:
        return self._price_with_params(market_parameters, option)

    def _current_values(self) -> dict[int, float]:
        return {u.underlying_id: u.value for u in self.underlying_state}

    def _price_with_params(self, params: MarketParameters, option: BinaryOption) -> float:
        vals: dict[int, float] = self._current_values()
        steps: int = option.steps_until_expiry

        if steps == 0:
            obs: float = option.observable_value(vals)
            return 1.0 if obs >= option.strike else 0.0

        rate_0: float = vals.get(FED_FUNDS_RATE_UNDERLYING_ID, 0.0)

        if len(option.legs) == 1:
            return self._price_single_leg(params, option, steps, vals, rate_0)
        return self._price_spread(params, option, steps, vals, rate_0)

    def _price_single_leg(
        self,
        params: MarketParameters,
        option: BinaryOption,
        steps: int,
        vals: dict[int, float],
        rate_0: float,
    ) -> float:
        leg: OptionLeg = option.legs[0]
        weight: float = leg.weight
        underlying_id: int = leg.underlying_id

        # --- FED rate: exact finite-state Markov chain over the rate grid. ------
        if underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
            dist: dict[float, float] = self._rate_distribution(params, rate_0, steps)
            threshold: float = option.strike / weight
            if weight > 0:
                prob = sum(p for r, p in dist.items() if r >= threshold)
            else:
                prob = sum(p for r, p in dist.items() if r <= threshold)
            return self._clamp01(prob)

        # --- Company valuation: conditionally log-normal given the rate path. ---
        if underlying_id in (AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID):
            if underlying_id == AJARAI_UNDERLYING_ID:
                drift: float = params.ajarai_drift
                beta: float = params.ajarai_rate_beta
                log_v0: float = math.log(max(vals.get(AJARAI_UNDERLYING_ID, 1.0), 1e-300))
                var_step: float = (
                    params.ajarai_sector_beta ** 2 * params.sector_std_dev ** 2
                    + params.ajarai_idio_std_dev ** 2
                )
            else:
                drift = params.theriodic_drift
                beta = params.theriodic_rate_beta
                log_v0 = math.log(max(vals.get(THERIODIC_UNDERLYING_ID, 1.0), 1e-300))
                var_step = (
                    params.theriodic_sector_beta ** 2 * params.sector_std_dev ** 2
                    + params.theriodic_idio_std_dev ** 2
                )

            sigma: float = math.sqrt(max(var_step * steps, 0.0))
            dist = self._rate_distribution(params, rate_0, steps)
            threshold = option.strike / weight
            prob = 0.0
            for r, rp in dist.items():
                mu: float = log_v0 + steps * drift + beta * (r - rate_0)
                if weight > 0:
                    if threshold <= 0:
                        c = 1.0
                    elif sigma <= 0:
                        c = 1.0 if mu >= math.log(threshold) else 0.0
                    else:
                        c = self._norm_cdf((mu - math.log(threshold)) / sigma)
                else:
                    # value must be <= a non-positive number: impossible.
                    if threshold <= 0:
                        c = 0.0
                    elif sigma <= 0:
                        c = 1.0 if mu <= math.log(threshold) else 0.0
                    else:
                        c = self._norm_cdf((math.log(threshold) - mu) / sigma)
                prob += rp * c
            return self._clamp01(prob)

        return self._monte_carlo_price(params, option, steps, vals, rate_0)

    def _price_spread(
        self,
        params: MarketParameters,
        option: BinaryOption,
        steps: int,
        vals: dict[int, float],
        rate_0: float,
    ) -> float:
        legs: tuple[OptionLeg, ...] = option.legs
        a_legs = [l for l in legs if l.underlying_id == AJARAI_UNDERLYING_ID]
        t_legs = [l for l in legs if l.underlying_id == THERIODIC_UNDERLYING_ID]

        # Only the "which valuation is higher" spread is tractable analytically.
        if len(legs) != 2 or len(a_legs) != 1 or len(t_legs) != 1:
            return self._monte_carlo_price(params, option, steps, vals, rate_0)

        w_a: float = a_legs[0].weight
        w_t: float = t_legs[0].weight

        if option.strike != 0.0 or w_a * w_t >= 0:
            return self._monte_carlo_price(params, option, steps, vals, rate_0)

        log_a: float = math.log(max(vals.get(AJARAI_UNDERLYING_ID, 1.0), 1e-300))
        log_t: float = math.log(max(vals.get(THERIODIC_UNDERLYING_ID, 1.0), 1e-300))
        sector_var: float = params.sector_std_dev ** 2

        if w_a > 0:
            # D = log(AJR) - log(THR); condition D >= log(-w_t / w_a).
            drift_diff = params.ajarai_drift - params.theriodic_drift
            beta_diff = params.ajarai_rate_beta - params.theriodic_rate_beta
            mu0 = log_a - log_t
            var_diff = (
                (params.ajarai_sector_beta - params.theriodic_sector_beta) ** 2 * sector_var
                + params.ajarai_idio_std_dev ** 2
                + params.theriodic_idio_std_dev ** 2
            )
            c = math.log(-w_t / w_a)
        else:
            drift_diff = params.theriodic_drift - params.ajarai_drift
            beta_diff = params.theriodic_rate_beta - params.ajarai_rate_beta
            mu0 = log_t - log_a
            var_diff = (
                (params.theriodic_sector_beta - params.ajarai_sector_beta) ** 2 * sector_var
                + params.theriodic_idio_std_dev ** 2
                + params.ajarai_idio_std_dev ** 2
            )
            c = math.log(-w_a / w_t)

        sigma = math.sqrt(max(var_diff * steps, 0.0))
        dist = self._rate_distribution(params, rate_0, steps)
        prob = 0.0
        for r, rp in dist.items():
            mu = mu0 + steps * drift_diff + beta_diff * (r - rate_0)
            if sigma <= 0:
                cc = 1.0 if mu >= c else 0.0
            else:
                cc = self._norm_cdf((mu - c) / sigma)
            prob += rp * cc
        return self._clamp01(prob)

    # ------------------------------------------------------------ numeric utils
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    @staticmethod
    def _clamp01(x: float) -> float:
        if x != x:  # NaN
            return 0.5
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _rate_distribution(
        self, params: MarketParameters, rate_0: float, steps: int
    ) -> dict[float, float]:
        dist: dict[float, float] = {round(rate_0, 2): 1.0}
        for _ in range(steps):
            new_dist: defaultdict[float, float] = defaultdict(float)
            for r, p in dist.items():
                up_p, down_p = params.tilted_rate_probabilities(r)
                stay_p = 1.0 - up_p - down_p
                if up_p > 0:
                    new_dist[params.next_rate_value(r, 1)] += p * up_p
                if down_p > 0:
                    new_dist[params.next_rate_value(r, -1)] += p * down_p
                if stay_p > 0:
                    new_dist[r] += p * stay_p
            dist = dict(new_dist)
        return dist

    def _monte_carlo_price(
        self,
        params: MarketParameters,
        option: BinaryOption,
        steps: int,
        vals: dict[int, float],
        rate_0: float,
    ) -> float:
        # Deterministic local RNG so price_option stays side-effect free.
        rng: random.Random = random.Random(option.option_id * 1000003 + 17)
        hits: int = 0
        samples: int = 20000
        for _ in range(samples):
            cur: dict[int, float] = dict(vals)
            for _ in range(steps):
                cur = self._advance_step_rng(params, cur, rng)
            if option.observable_value(cur) >= option.strike:
                hits += 1
        return self._clamp01(hits / samples)

    def _advance_step_rng(
        self, params: MarketParameters, cur: dict[int, float], rng: random.Random
    ) -> dict[int, float]:
        current_rate: float = cur[FED_FUNDS_RATE_UNDERLYING_ID]
        up_p, down_p = params.tilted_rate_probabilities(current_rate)
        draw: float = rng.random()
        if draw < up_p:
            rate_value: float = params.next_rate_value(current_rate, 1)
        elif draw < up_p + down_p:
            rate_value = params.next_rate_value(current_rate, -1)
        else:
            rate_value = current_rate
        rate_change: float = round(rate_value - current_rate, 2)
        sector_shock: float = rng.gauss(0.0, params.sector_std_dev)

        def company(
            cur_val: float,
            drift: float,
            rate_beta: float,
            sector_beta: float,
            idio_std_dev: float,
        ) -> float:
            idio_shock: float = rng.gauss(0.0, idio_std_dev)
            log_return: float = drift + rate_beta * rate_change + sector_beta * sector_shock + idio_shock
            return round(cur_val * math.exp(log_return), 2)

        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: company(
                cur[AJARAI_UNDERLYING_ID],
                params.ajarai_drift,
                params.ajarai_rate_beta,
                params.ajarai_sector_beta,
                params.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: company(
                cur[THERIODIC_UNDERLYING_ID],
                params.theriodic_drift,
                params.theriodic_rate_beta,
                params.theriodic_sector_beta,
                params.theriodic_idio_std_dev,
            ),
        }

    # ------------------------------------------------------------------- trading
    def _risk_budget(self) -> float:
        return max(0.0, self.cash_balance - self._cash_floor)

    def _bid_quantity(self, bid_price: float) -> int:
        budget: float = min(self._quote_max_loss, self._risk_budget())
        if bid_price <= 0:
            qty = self.MAX_QTY
        else:
            qty = int(budget / max(bid_price, 1e-9))
        return max(1, min(qty, self.MAX_QTY))

    def _offer_quantity(self, offer_price: float) -> int:
        budget: float = min(self._quote_max_loss, self._risk_budget())
        max_loss_per_contract: float = 1.0 - offer_price
        if max_loss_per_contract <= 0:
            qty = self.MAX_QTY
        else:
            qty = int(budget / max(max_loss_per_contract, 1e-9))
        return max(1, min(qty, self.MAX_QTY))

    @staticmethod
    def _floor_penny(x: float) -> float:
        return math.floor(x * 100.0) / 100.0

    @staticmethod
    def _ceil_penny(x: float) -> float:
        return math.ceil(x * 100.0) / 100.0

    def _skew_adjusted_fair(self, option: BinaryOption, fair: float) -> float:
        # Shift fair value against our net inventory so we lean toward flattening:
        # when long we quote a touch low (eager to sell, reluctant to buy), when
        # short the reverse. The shift is scaled to one full quote's size, so it
        # is invariant to the session's cash/quantity scale.
        net: int = self.position.option_quantity_by_option_id.get(option.option_id, 0)
        if net == 0:
            return fair
        # Clamp the reference fair before sizing: the raw fair can be a denormal
        # (or exactly 0/1), which would overflow the quantity formulas above.
        fair_ref: float = min(max(fair, 0.01), 0.99)
        ref: float = float(max(self._bid_quantity(fair_ref), self._offer_quantity(fair_ref)))
        skew: float = self.HALF_EDGE * (net / max(ref, 1.0))
        skew = max(-self.MAX_SKEW, min(self.MAX_SKEW, skew))
        return self._clamp01(fair - skew)

    def quote(self, option: BinaryOption, counterparty_id: int) -> Quote:
        # Once we've spent our risk budget, stop taking on exposure. Quote an
        # unfillable two-sided market that still cannot lose money if hit.
        if self._risk_budget() <= 0:
            return Quote(bid_price=0.0, bid_quantity=self.MAX_QTY, offer_price=1.0, offer_quantity=self.MAX_QTY)

        fair: float = self._skew_adjusted_fair(option, self.price_option(option))
        bid: float = min(max(self._floor_penny(fair - self.HALF_EDGE), 0.0), 1.0)
        offer: float = min(max(self._ceil_penny(fair + self.HALF_EDGE), 0.0), 1.0)

        # Guarantee a valid two-sided market even at the extremes.
        if bid >= offer:
            if offer <= 0.01:
                bid = 0.0
                offer = 0.01
            else:
                bid = max(0.0, offer - 0.01)
                if bid >= offer:
                    bid = 0.0
                    offer = 0.01

        return Quote(
            bid_price=bid,
            bid_quantity=self._bid_quantity(bid),
            offer_price=offer,
            offer_quantity=self._offer_quantity(offer),
        )

    def respond_to_fok(self, option: BinaryOption, fok_order: FokOrder) -> bool:
        fair: float = self._skew_adjusted_fair(option, self.price_option(option))
        price: float = fok_order.price
        qty: int = fok_order.quantity

        if fok_order.order_type == OrderType.BUY:
            # Counterparty buys -> we sell at `price`. Accept any sale that clears
            # our inventory-adjusted fair by a minimal edge (adverse-selection guard).
            if price < fair + self.FOK_MIN_EDGE:
                return False
            max_loss: float = (1.0 - price) * qty
        else:
            # Counterparty sells -> we buy at `price`.
            if price > fair - self.FOK_MIN_EDGE:
                return False
            max_loss = price * qty

        return max_loss <= self._fok_max_loss and max_loss <= self._risk_budget() + 1e-9

    # --------------------------------------------------------------- estimation
    def warm_up(self, market_history: MarketHistory) -> None:
        try:
            self._params = self._estimate_params(market_history)
        except Exception:
            self._params = self._default_params()

    def _estimate_params(self, history: MarketHistory) -> MarketParameters:
        values_by_id: dict[int, tuple[float, ...]] = history.values_by_underlying_id
        rate_hist: tuple[float, ...] | None = values_by_id.get(FED_FUNDS_RATE_UNDERLYING_ID)
        a_hist: tuple[float, ...] | None = values_by_id.get(AJARAI_UNDERLYING_ID)
        t_hist: tuple[float, ...] | None = values_by_id.get(THERIODIC_UNDERLYING_ID)

        if not rate_hist or len(rate_hist) < 3:
            return self._default_params()

        rate_changes: list[float] = [
            round(rate_hist[i] - rate_hist[i - 1], 2) for i in range(1, len(rate_hist))
        ]

        up, down, rev, target = self._estimate_rate_params(rate_hist, rate_changes)

        if a_hist is None or t_hist is None or len(a_hist) < 3 or len(t_hist) < 3:
            return self._default_params()

        a_drift, a_beta, a_var, t_drift, t_beta, t_var, cov = self._estimate_company_params(
            a_hist, t_hist, rate_changes
        )

        sector_std, beta_a, beta_t, idio_a, idio_t = self._decompose(a_var, t_var, cov)

        # Light clipping to keep the model numerically sane.
        a_drift = max(-0.5, min(0.5, a_drift))
        t_drift = max(-0.5, min(0.5, t_drift))
        a_beta = max(-10.0, min(10.0, a_beta))
        t_beta = max(-10.0, min(10.0, t_beta))
        idio_a = max(0.0, min(1.0, idio_a))
        idio_t = max(0.0, min(1.0, idio_t))
        sector_std = max(0.0, min(1.0, sector_std))

        return MarketParameters(
            ajarai_drift=a_drift,
            ajarai_idio_std_dev=idio_a,
            ajarai_rate_beta=a_beta,
            ajarai_sector_beta=beta_a,
            rate_down_probability=down,
            rate_reversion_strength=rev,
            rate_up_probability=up,
            sector_std_dev=sector_std,
            theriodic_drift=t_drift,
            theriodic_idio_std_dev=idio_t,
            theriodic_rate_beta=t_beta,
            theriodic_sector_beta=beta_t,
            rate_step=self.RATE_STEP,
            rate_target=target,
        )

    def _estimate_rate_params(
        self, rate_hist: tuple[float, ...], rate_changes: list[float]
    ) -> tuple[float, float, float, float]:
        transitions: list[tuple[float, float]] = [
            (rate_hist[i - 1], rate_changes[i - 1]) for i in range(1, len(rate_hist))
        ]

        def loglik(up: float, down: float, rev: float, target: float) -> float:
            total = 0.0
            for r_prev, d in transitions:
                tilt = rev * (target - r_prev)
                up_p = min(max(up + tilt, 0.0), 1.0)
                down_p = min(max(down - tilt, 0.0), 1.0 - up_p)
                if d > 0:
                    total += math.log(max(up_p, 1e-12))
                elif d < 0:
                    total += math.log(max(down_p, 1e-12))
                else:
                    # At rate 0 a "down" draw is clamped to 0, so staying has prob 1 - up.
                    stay_p = 1.0 - up_p if r_prev <= 1e-9 else 1.0 - up_p - down_p
                    total += math.log(max(stay_p, 1e-12))
            return total

        up_candidates = [0.05 * i for i in range(1, 19)]
        down_candidates = [0.05 * i for i in range(1, 19)]
        rev_candidates = [0.05 * i for i in range(0, 21)]
        target_candidates = [0.5 * i for i in range(1, 9)]

        best = (0.30, 0.30, 0.10, 2.0)
        best_ll = loglik(*best)

        # Coarse coordinate ascent.
        for _ in range(4):
            for idx, cands in (
                (0, up_candidates),
                (1, down_candidates),
                (2, rev_candidates),
                (3, target_candidates),
            ):
                cur = list(best)
                for c in cands:
                    cur[idx] = c
                    up, down, rev, target = cur
                    if up <= 0 or down <= 0 or up + down > 1.0:
                        continue
                    if rev < 0 or rev > 1.0 or target < 0:
                        continue
                    v = loglik(up, down, rev, target)
                    if v > best_ll:
                        best_ll = v
                        best = (up, down, rev, target)

        # Fine local refinement.
        for _ in range(3):
            for idx in range(4):
                center = best[idx]
                cur = list(best)
                for k in range(-10, 11):
                    cur[idx] = center + 0.005 * k
                    up, down, rev, target = cur
                    if up <= 0 or down <= 0 or up + down > 1.0:
                        continue
                    if rev < 0 or rev > 1.0 or target < 0:
                        continue
                    v = loglik(up, down, rev, target)
                    if v > best_ll:
                        best_ll = v
                        best = (up, down, rev, target)

        return best

    def _estimate_company_params(
        self,
        a_hist: tuple[float, ...],
        t_hist: tuple[float, ...],
        rate_changes: list[float],
    ) -> tuple[float, float, float, float, float, float, float]:
        a_logs = [math.log(max(v, 1e-300)) for v in a_hist]
        t_logs = [math.log(max(v, 1e-300)) for v in t_hist]
        n = len(rate_changes)
        if n < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        a_ret = [a_logs[i + 1] - a_logs[i] for i in range(n)]
        t_ret = [t_logs[i + 1] - t_logs[i] for i in range(n)]

        def ols(y: list[float], x: list[float]) -> tuple[float, float, list[float]]:
            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)
            sxx = sum((xi - mean_x) ** 2 for xi in x)
            sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            beta = sxy / sxx if sxx > 1e-12 else 0.0
            drift = mean_y - beta * mean_x
            resid = [yi - drift - beta * xi for xi, yi in zip(x, y)]
            return drift, beta, resid

        a_drift, a_beta, a_resid = ols(a_ret, rate_changes)
        t_drift, t_beta, t_resid = ols(t_ret, rate_changes)

        m = len(a_resid)
        a_var = sum(r * r for r in a_resid) / m
        t_var = sum(r * r for r in t_resid) / m
        cov = sum(ra * rb for ra, rb in zip(a_resid, t_resid)) / m
        return a_drift, a_beta, a_var, t_drift, t_beta, t_var, cov

    def _decompose(
        self, a_var: float, t_var: float, cov: float
    ) -> tuple[float, float, float, float, float]:
        """Turn residual variance/covariance into a valid MarketParameters
        decomposition. Only the combinations used by the pricing formulas
        (single-leg variance and the difference variance) matter."""
        tiny = 1e-16
        if a_var <= tiny and t_var <= tiny:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        if a_var >= t_var:
            sigma2 = a_var
            beta_a = 1.0
            beta_t = cov / a_var if a_var > tiny else 0.0
            idio_a2 = 0.0
            idio_t2 = t_var - (cov * cov) / a_var if a_var > tiny else t_var
        else:
            sigma2 = t_var
            beta_t = 1.0
            beta_a = cov / t_var if t_var > tiny else 0.0
            idio_t2 = 0.0
            idio_a2 = a_var - (cov * cov) / t_var if t_var > tiny else a_var

        return (
            math.sqrt(max(sigma2, 0.0)),
            beta_a,
            beta_t,
            math.sqrt(max(idio_a2, 0.0)),
            math.sqrt(max(idio_t2, 0.0)),
        )