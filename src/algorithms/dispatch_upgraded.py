"""A dispatch system built on the starter. HANDBOOK.md says what the starter does; RULES.md is the law.

What changed from the starter, and why (every change answers something the handbook or the rules say):

 1. It no longer wins every car. The starter quotes the list price on all requests, wins what it cannot move,
    and falls under the 90% delivery minimum. Now each quote is the list price times a markup, never below an
    estimate of what the car will cost to move. A quote that is too dear simply loses, which costs nothing.
 2. The estimate uses where the drivers live (the driver list and the map), how many cars one dealer
    group has sent in one go (bulk orders), and what the note says.
 3. It watches itself. Warnings, failed visits, no-shows and late refunds raise its prices for a while, so it
    takes fewer cars exactly when it is failing to move them. Booking fees teach it what its fees must be.
 4. Fees follow the time left. An order that is already inside its 3-hour warning is posted at the steep fees
    at once, and a gig that a driver dropped starts its climb again from the fee it had reached.
 5. At a deadline warning it also sends direct offers to the nearest fitting drivers.
 6. It reads notes as RULES.md says: days count from the day the request arrives, weekday names, "tomorrow",
    am/pm, hours in words, "half three", a correction counts, a denied time does not. It also reads
    manual gearboxes and missing plates, so those gigs go out right the first time.

Added in this version (each answers a sentence of RULES.md or HANDBOOK.md):

 7. It learns the price level the dealers accept (HANDBOOK, "Dealers compare prices": "a system that keeps learning from
    its own wins and losses keeps its margin"). `Market` moves a multiple of the list price up after a win and down
    after a loss, in the ratio that makes LEARN_TARGET of the clean quotes win. A small deviation for each pickup place
    that fades in a day catches the outage (a backup 30% dearer) and the price war (20% cheaper) without a script.
    Only a quote that was priced at the market level teaches: a loss at a price that was high on purpose says nothing.
 8. Dealer trust (level 8): each on-time order raises the price a dealer accepts by 1%, up to 25; a late order halves the
    count. The desk counts it from the ledger, as RULES.md says, and quotes that much higher to that dealer.
 9. A guard for the delivery minimum. Being under 90% on time means no rank at all, and a quote that loses costs nothing.
    When the late share, plus the orders at risk, climbs near the limit, same-day and risky requests are declined first.
10. The cost estimate follows the rules' list of what a driver dislikes: private addresses, evenings, Saturdays,
    trade-ins and charging stops (the typical fee is the drive and the trips only). A same-day option costs more to
    serve than a three-day one. A bulk order is judged against the drivers who live near its yard.
11. Cars that sit in the app without a driver raise the price of new work (the shortage shows there first, before the
    first deadline warning). Gig fees follow the scarcity of drivers around the pickup, and the fee cap lifts for
    an order inside its warning and more for one that is late, because an undelivered car costs twice its price.

All the numbers in the TUNING block are educated guesses, not measurements. Run the same seeds before and
after you change one, and look at the margin AND the on-time share. To run it: python dispatch.py --level 8
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from typing import Any, NamedTuple, Optional

from spread.client import Client, SpreadError, Unreachable, World
from spread.types import (
    Action, ActionResult, Audience, DeskEvent, DeskEventKind, DeskSnapshot, FeeStep, Gig, Note, Order, PostAction, PriceRequest,
    Score, StopRef, TimeWindow,
)

HOUR, DAY = 60, 24 * 60
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# The events that need an answer. Views and account lines arrive with the next event that needs an answer.
WAKE_ON: list[DeskEventKind] = ["price_request", "order_won", "order_changed", "order_cancelled", "visit_failed", "gig_expired", "deadline_close",
                                "gig_dropped", "no_show"]

# --- TUNING: guesses, to be swept with --seed ---------------------------------------------------------------------------
# The quote is: list price * what the market teaches * trust * OPTION_FACTOR * what trouble adds, but never under the estimated
# cost * OPTION_COST. The competitor is about 10% either way of the list at levels up to 6, 12-38% above it at level 7, 26-54%
# above at level 8. MARKUP is only where the market starts: Market then learns the level from the dealers' answers.
MARKUP = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.03, 7: 1.10, 8: 1.18}
# Reserve more margin for urgent work and compete more strongly when the dealer gives us time.
OPTION_FACTOR = {"same_day": 1.18, "next_day": 1.04, "within_3_days": 0.92}
OPTION_COST = {"same_day": 1.22, "next_day": 1.02, "within_3_days": 0.90}
LEARN_TARGET = {4: 0.55, 5: 0.55, 6: 0.50, 7: 0.40, 8: 0.35}  # the share of clean quotes to win; fewer where drivers are few
LEARN_STEP = 0.03  # the size of one lesson, in log units: a win teaches (1 - target) of it, a loss target of it
LEARN_CLEAN = 1.04  # a quote teaches only when it was at most 4% over the market level, trust apart
LEARN_SHARE = 0.5  # the part of a lesson that goes to the whole market; the rest goes to the pickup place
LEVEL_LIMITS = (0.80, 2.0)  # the market level, as a multiple of the list price
PLACE_LIMITS, PLACE_HALF_LIFE = (-0.35, 0.45), 24 * HOUR  # a place's deviation: an outage or a price war lasts one day
TRUST_STEP, TRUST_MAX = 0.01, 25  # RULES.md, dealer trust at level 8
GUARD_SOFT, GUARD_HARD = 0.05, 0.075  # at-risk share of the orders won: decline same-day and risky work, then nearly all new work
GUARD_SCARCITY, GUARD_SCARCITY_HARD, GUARD_RISK = 1.35, 1.15, 1.15
DECLINE_MULT = 6.0  # a declining quote: this many times the list price, far above any dealer's limit
EXTRA_PRIVATE, EXTRA_EVENING, EXTRA_WEEKEND, EXTRA_TRADE_IN, EXTRA_CHARGING = 0.08, 0.10, 0.20, 0.04, 0.05  # what drivers dislike
EVENING_FROM = 17 * HOUR  # a pickup from this hour on is evening work
STALE_AFTER = 90  # minutes in the app without a booking before an order counts as stale
BACKLOG_FREE, BACKLOG_SLOPE, BACKLOG_CAP = 0.06, 0.02, 0.5  # stale orders above 6% of the drivers raise the price 2% each, up to 50%
BULK_FREE_MIN, BULK_FREE_MAX, BULK_SUPPLY_SHARE = 4, 12, 0.5  # cars from one yard that cost no extra: half the drivers near it
BULK_SLOPE, BULK_CAP = 0.04, 1.0
FEE_SCARCITY = 0.4  # a gig's fees rise by this share of the pickup's scarcity premium
CAP_URGENT, CAP_LATE = 1.9, 2.2  # the fee cap, as a share of the price: inside the warning; late (an undelivered car costs 2x the price)
EXPECTED_FEE = {1: 1.1, 2: 1.1, 3: 1.1, 4: 1.1, 5: 1.1, 6: 1.15, 7: 1.4, 8: 1.5}  # fees paid, as a multiple of the typical fee
MARGIN = 1.08
REACH_MINUTES = 75  # a driver at home counts as near a place when the road takes at most this long
OFFER_REACH = 150  # direct offers go to drivers whose home is at most this far from the pickup
MIN_WIN_SHARE, EASE_MIN = 0.12, 0.55  # if under 12% of the requests become orders, the cost floor eases by 1% a request, down to 55%
PRESSURE_HALF_LIFE = 8 * HOUR  # trouble fades with this half-life
PRESSURE_SLOPE = 0.03  # each unit of trouble raises the markup by 3%, up to PRESSURE_CAP
PRESSURE_CAP = 1.5
LATE_FREE, LATE_WEIGHT = 0.03, 60.0  # late share above 3% adds trouble: 10% late adds 4.2 units
LOAD_FREE, LOAD_SLOPE = 2.0, 0.03  # open orders per driver above 2 raise the markup by 3% for each one more
FEE_MIN, FEE_MAX = 0.75, 1.6  # what booking fees can teach the starting fee: a multiple of the typical fee
FEE_CAP_SHARE = 1.5  # a gig's fee does not climb above 1.5 times its share of the order's price (but never under its first fee)
URGENT_LEAD = 3 * HOUR  # inside this time to `lastPickupAt` the steep fees apply at once, as after the warning


class Retry(NamedTuple):
    """How to post a leg again after a failed visit."""
    pickup_from: int = 0
    manual_licence: bool = False
    plates: bool = False


class Quoted(NamedTuple):
    """What the desk sent for a request: what each option would have cost at the market level, and the price that went out."""
    place: str
    trust: float
    units: dict
    prices: dict


class Market:
    """What the dealers' answers teach about the price they accept, as a multiple of the list price.

    `level` holds for the whole market. `shift` is a deviation for each pickup place, in log units, that fades with
    PLACE_HALF_LIFE: an outage or a price war lasts a day, while the lanes of a place last the week. A win teaches that
    the price may rise and a loss that it may fall. The lessons are in the ratio (1 - target) : target, so the share of
    the clean quotes that win settles at `target`. The caller teaches only with quotes that were priced at the level."""

    def __init__(self, prior: float, target: float):
        self.prior, self.level = prior, prior
        self.up, self.down = (1 - target) * LEARN_STEP, target * LEARN_STEP
        self.shift: dict[str, tuple[float, int]] = {}  # place -> (deviation, the minute it was set)

    def deviation(self, place: str, now: int) -> float:
        value, at = self.shift.get(place, (0.0, now))
        return value * 0.5 ** (max(0, now - at) / PLACE_HALF_LIFE)

    def multiple(self, place: str, now: int) -> float:
        return self.level * math.exp(self.deviation(place, now))

    def learn(self, place: str, now: int, won: bool) -> None:
        lesson = self.up if won else -self.down
        self.level = min(LEVEL_LIMITS[1], max(LEVEL_LIMITS[0], self.level * math.exp(lesson * LEARN_SHARE)))
        value = self.deviation(place, now) + lesson * (1 - LEARN_SHARE)
        self.shift[place] = (min(PLACE_LIMITS[1], max(PLACE_LIMITS[0], value)), now)


# --- Where the drivers live ------------------------------------------------------------------------------------------------

class Supply:
    """The driver list and the map: how many drivers live near a place, and who is nearest to it."""

    def __init__(self, drivers: list[dict[str, Any]], world_map: Optional[dict[str, Any]]):
        self.drivers = drivers
        self.minutes: dict[tuple[str, str], int] = {}
        self.km: dict[tuple[str, str], float] = {}
        places: list[str] = []
        if world_map:
            self.minutes = {(r["from"], r["to"]): r["minutes"] for r in world_map["roads"]}
            self.km = {(r["from"], r["to"]): r["km"] for r in world_map["roads"]}
            places = [p["id"] for p in world_map["places"]]
        homes = Counter(d["homePlaceId"] for d in drivers)
        experienced = Counter(d["homePlaceId"] for d in drivers if experienced_driver(d))
        self.near = {p: self.within(homes, p) for p in places}
        self.near_experienced = {p: self.within(experienced, p) for p in places}
        self.typical = median([n for n in self.near.values() if n > 0])
        self.typical_experienced = median([n for n in self.near_experienced.values() if n > 0])
        self._nearest: dict[str, list[dict[str, Any]]] = {}

    @staticmethod
    def empty() -> "Supply":
        return Supply([], None)

    def within(self, homes: Counter, place: str) -> int:
        return sum(n for home, n in homes.items() if self.minutes.get((home, place), 10 ** 6) <= REACH_MINUTES)

    def scarcity(self, pickup: str, delivery: str, premium: bool) -> float:
        """1.0 where drivers are plentiful, up to about 2 where few live near the pickup and the delivery."""
        if not self.typical:
            return 1.0
        pick = min(1.0, self.near.get(pickup, 0) / self.typical)
        back = min(1.0, self.near.get(delivery, 0) / self.typical)  # a driver who ends near home is cheap
        factor = 1 + 0.6 * (1 - pick) + 0.25 * (1 - back)
        if premium and self.typical_experienced:
            factor += 0.3 * (1 - min(1.0, self.near_experienced.get(pickup, 0) / self.typical_experienced))
        return factor

    def distance_km(self, a: str, b: str) -> Optional[float]:
        return self.km.get((a, b))

    def nearest(self, place: str, reach: int) -> list[dict[str, Any]]:
        """The drivers whose home is within `reach` minutes of the place: nearest first, fewest drops first."""
        if place not in self._nearest:
            close = [(self.minutes.get((d["homePlaceId"], place), 10 ** 6), d["drops"], d) for d in self.drivers]
            self._nearest[place] = [d for m, _, d in sorted(close, key=lambda c: (c[0], c[1])) if m <= reach]
        return self._nearest[place]


def experienced_driver(driver: dict[str, Any]) -> bool:
    return driver["rating"] >= 4.8 and driver["completedGigs"] >= 50


def median(values: list[int]) -> float:
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2]) if ordered else 0.0


def load_supply(world: World) -> Supply:
    """The driver list and the map. A server that is gone is for the caller to handle; any other failure gives a neutral supply."""
    try:
        return Supply(world.roster(), world.client.map())
    except Unreachable:
        raise
    except SpreadError:
        return Supply.empty()


# --- The desk --------------------------------------------------------------------------------------------------------------

class Desk:
    """What the system remembers, and its answer to each event. The answers wait in the outbox
    until the system sends them, and the world applies them at the current minute."""

    def __init__(self, world: World, level: int, now: int, supply: Optional[Supply] = None):
        self.world, self.level, self.now = world, level, now
        self.supply = supply or Supply.empty()
        self.outbox: list[Action] = []  # the actions decided since the desk last sent
        self.requests: dict[str, PriceRequest] = {}
        self.orders: dict[str, Order] = {}  # the orders won and not cancelled
        self.leg_gigs: dict[str, dict[int, str]] = {}  # order id -> leg -> the gig that carries the leg
        self.booked: set[str] = set()  # the gigs that a driver holds now
        self.posts: dict[str, PostAction] = {}  # gig id -> the post that made it
        self.schedules: dict[str, list[FeeStep]] = {}  # gig id -> its fee schedule now
        self.urgent: set[str] = set()  # the orders whose deadline warning has come
        self.late: set[str] = set()  # the orders with a late refund
        self.batches: Counter = Counter()  # batch id -> the requests seen from that dealer group
        self.asked, self.took, self.ease = 0, 0, 1.0  # requests seen, orders won, and how much of the cost floor still applies
        self.fee_mult = 1.0  # what bookings have taught: the starting fee, as a multiple of the typical fee
        self.pressure, self.pressure_at = 0.0, now  # trouble, fading with time
        self.offered: dict[str, set[str]] = {}  # gig id -> the drivers who got an offer for it
        self.offers_by_day: Counter = Counter()  # (driver id, day) -> offers sent: at most three
        self.market = Market(MARKUP[level], LEARN_TARGET.get(level, 0.5))  # the price level that the dealers accept
        self.drafts: dict[str, Quoted] = {}  # request id -> the quote in the outbox, until the world accepts it
        self.quoted: dict[str, Quoted] = {}  # request id -> the quote that the world accepted, until the dealer decides
        self.trust: Counter = Counter()  # dealer id -> its count of your on-time orders (level 8)
        self.counted: set[str] = set()  # the orders counted on time
        self.refunded: set[str] = set()  # the orders that had a late refund
        self.first_posted: dict[str, int] = {}  # order id -> the minute its first gig reached the app
        self.batch_wins: Counter = Counter()  # batch id -> the orders won from that dealer group

    # --- Starting again ---

    def rejoin(self, snapshot: DeskSnapshot, history: list[DeskEvent]) -> None:
        """Rebuilds what it knows from the world, so a desk that stopped keeps its orders. Then answers
        what it missed: open requests, orders without a gig, failed visits without a new gig, and the
        deadline warnings, so the orders that had one keep their steep fees."""
        posted: dict[str, list[Gig]] = {}  # order id -> its gigs, oldest first
        self.booked = {gig["id"] for gig in snapshot["gigs"] if gig["status"] in ("booked", "working", "done")}
        for gig in snapshot["gigs"]:
            posted.setdefault(gig["stops"][0]["orderId"], []).append(gig)
            if gig["status"] in ("open", "booked", "working"):
                self.carry({"type": "post", "stops": gig["stops"], "audience": gig["audience"], "schedule": gig["schedule"]}, gig["id"])
        self.requests = {e["request"]["id"]: e["request"] for e in history if e["kind"] == "price_request"}
        self.urgent = {e["orderId"] for e in history if e["kind"] == "deadline_close"}
        self.late = {e["line"]["orderId"] for e in history if e["kind"] == "ledger" and e["line"]["kind"] == "late_refund" and e["line"]["orderId"]}
        for request in self.requests.values():
            if request["batchId"]:
                self.batches[request["batchId"]] += 1
        for event in history:
            if event["kind"] == "ledger":
                self.trust_line(event["line"])
            elif event["kind"] == "order_won" and event["order"]["id"] in self.requests:
                batch = self.requests[event["order"]["id"]]["batchId"]
                if batch:
                    self.batch_wins[batch] += 1
        self.orders = {o["id"]: o for o in snapshot["orders"] if o["id"] in self.requests and o["status"] != "cancelled"}
        for order in self.orders.values():
            if order["status"] == "waiting" and not self.gigs_of(order["id"]):
                self.post(order, undelivered(order))
        for event in history:
            order = self.orders.get(event.get("orderId", ""))
            if order is None:
                continue
            if event["kind"] == "visit_failed" and order["deliveredAt"][event["leg"]] is None:
                later = [g for g in posted.get(order["id"], []) if g["postedAt"] >= event["at"] and g["id"] != event["gigId"]]
                if not any(stop["leg"] == event["leg"] for gig in later for stop in gig["stops"]):
                    self.retry(order["id"], event["leg"], event["reason"])
        for request in snapshot["openRequests"]:
            self.quote(request)

    # --- Events ---

    def handle(self, event: DeskEvent) -> None:
        kind = event["kind"]
        if kind == "price_request":
            self.quote(event["request"])
        elif kind == "order_won":
            self.win(event["order"])
        elif kind == "order_lost":  # the dealer took the competitor's price: ours was above what it accepts today
            self.learn_outcome(event["requestId"], False, None)
        elif kind == "order_changed":
            self.change(event["order"])
        elif kind == "order_cancelled":  # the world has stopped the car and told each of its gigs what happens
            for books in (self.orders, self.leg_gigs):
                books.pop(event["orderId"], None)
            self.late.discard(event["orderId"])
        elif kind == "visit_failed":
            self.worry(0.5)
            self.retry(event["orderId"], event["leg"], event["reason"])
        elif kind == "gig_expired":
            self.worry(1.0)
            self.fee_mult = min(FEE_MAX, self.fee_mult * 1.03)
            self.repost(event["gigId"])
        elif kind == "deadline_close":  # it comes again when a driver drops the car after the first warning
            self.raise_fees(event["orderId"])
        elif kind == "gig_booked":
            self.booked.add(event["gigId"])
            self.learn_booking(event["gigId"], event["feeCents"])
        elif kind in ("gig_dropped", "no_show"):  # after a no-show the gig is open again; a late driver still comes
            self.booked.discard(event["gigId"])
            self.worry(0.3 if kind == "gig_dropped" else 0.5)
            self.revive(event["gigId"])
        elif kind == "delivered":
            self.delivered(event["orderId"], event["leg"], event["at"])
        elif kind == "ledger":
            line = event["line"]
            self.trust_line(line)
            if line["kind"] == "late_refund" and line["orderId"] and line["orderId"] not in self.late:
                self.late.add(line["orderId"])
                self.worry(2.0)

    def delivered(self, order_id: str, leg: int, at: int) -> None:
        self.leg_gigs.get(order_id, {}).pop(leg, None)
        if order_id in self.orders:
            self.orders[order_id]["deliveredAt"][leg] = at

    # --- Trouble, and what it does to the prices ---

    def strain(self) -> float:
        return self.pressure * 0.5 ** (max(0, self.now - self.pressure_at) / PRESSURE_HALF_LIFE)

    def worry(self, amount: float) -> None:
        self.pressure, self.pressure_at = self.strain() + amount, self.now

    def late_share(self) -> float:
        """The orders known late, and those past their deadline with a car undelivered, over the orders won."""
        if not self.orders:
            return 0.0
        late = sum(1 for order_id in self.orders if order_id in self.late)
        overdue = sum(1 for o in self.orders.values() if o["id"] not in self.late and o["deadline"] < self.now and undelivered(o))
        return (late + overdue) / (len(self.orders) + 20)  # 20 orders on time as a prior: one late order in the first hour is not 100% late

    def pressure_factor(self) -> float:
        """What trouble, load and cars without a driver add to a price: 1.0 when all is well. The market level is not in it."""
        trouble = self.strain() + LATE_WEIGHT * max(0.0, self.late_share() - LATE_FREE)
        factor = min(PRESSURE_CAP, 1 + PRESSURE_SLOPE * trouble)
        if self.supply.drivers:
            drivers = len(self.supply.drivers)
            active = sum(1 for o in self.orders.values() if undelivered(o))
            factor *= 1 + LOAD_SLOPE * max(0.0, active / drivers - LOAD_FREE)
            stale = self.stale_orders()
            factor *= 1 + min(BACKLOG_CAP, BACKLOG_SLOPE * max(0.0, stale - max(2.0, BACKLOG_FREE * drivers)))
        return factor

    def stale_orders(self) -> int:
        """The orders with a gig in the app that no driver has booked for STALE_AFTER minutes: the shortage, before any warning."""
        stale = 0
        for order_id, order in self.orders.items():
            gigs = self.gigs_of(order_id)
            if gigs and undelivered(order) and not any(g in self.booked for g in gigs) \
                    and self.now - self.first_posted.get(order_id, self.now) > STALE_AFTER:
                stale += 1
        return stale

    def at_risk_share(self) -> float:
        """Late orders, orders past their deadline with a car undelivered, and half of the orders inside their deadline warning
        with no driver, over the orders won. The delivery minimum is 90% on time, so this is the number to keep low."""
        if not self.orders:
            return 0.0
        late = sum(1 for order_id in self.orders if order_id in self.late)
        overdue = sum(1 for o in self.orders.values() if o["id"] not in self.late and o["deadline"] < self.now and undelivered(o))
        waiting = sum(1 for order_id in self.urgent if order_id in self.orders and order_id not in self.late
                      and undelivered(self.orders[order_id]) and not any(g in self.booked for g in self.gigs_of(order_id)))
        return (late + overdue + 0.5 * waiting) / (len(self.orders) + 20)

    def trust_boost(self, dealer: str) -> float:
        """Level 8: a dealer accepts 1% more for each of your orders that it counted on time, up to 25."""
        return 1 + TRUST_STEP * self.trust[dealer] if self.level >= 8 else 1.0

    def trust_line(self, line: Any) -> None:
        """Counts a dealer's on-time orders from the ledger, as RULES.md says: a revenue line with no late refund is an order
        on time, and a late order halves the dealer's count."""
        order_id = line["orderId"]
        dealer = self.requests[order_id]["dealerId"] if order_id in self.requests else None
        if dealer is None:
            return
        if line["kind"] == "revenue" and order_id not in self.refunded and order_id not in self.counted:
            self.counted.add(order_id)
            self.trust[dealer] = min(TRUST_MAX, self.trust[dealer] + 1)
        elif line["kind"] == "late_refund" and order_id not in self.refunded:
            self.refunded.add(order_id)
            if order_id in self.counted:  # the revenue line came first: take its count back before halving
                self.counted.discard(order_id)
                self.trust[dealer] = max(0, self.trust[dealer] - 1)
            self.trust[dealer] //= 2

    def learn_booking(self, gig_id: str, fee: int) -> None:
        """A gig that books at its first fee was posted too high, and one that needed a raise too low."""
        action = self.posts.get(gig_id)
        if action is None or self.level < 2 or action["stops"][0]["orderId"] in self.urgent:
            return
        opening = action["schedule"][0]["feeCents"]
        ratio = fee / opening if opening > 0 else 1.0
        step = 0.985 if ratio <= 1.001 else 1.02 if ratio >= 1.3 else 1.0
        self.fee_mult = min(FEE_MAX, max(FEE_MIN, self.fee_mult * step))

    # --- Quotes ---

    def quote(self, request: PriceRequest) -> None:
        if request["id"] not in self.requests:
            self.asked += 1
            if request["batchId"]:
                self.batches[request["batchId"]] += 1
            if self.asked >= 40:  # a floor that wins nothing is worse than none: a team that wins no order is not ranked
                if self.took / self.asked < MIN_WIN_SHARE:
                    self.ease = max(EASE_MIN, self.ease - 0.01)
                elif self.took / self.asked > 2 * MIN_WIN_SHARE:
                    self.ease = min(1.0, self.ease + 0.003)
        self.requests[request["id"]] = request
        prices = self.prices_for(request)
        if prices:
            self.outbox.append({"type": "quote", "requestId": request["id"], "prices": prices})

    def prices_for(self, request: PriceRequest) -> dict[str, int]:
        """A price for each option: the list price at the market level, with what trust and trouble add, never under the cost of
        the option. {} declines: up to level 3, no quote; from level 4, a price so high that it loses is also a decline, and
        so is the guard's."""
        cost, scarcity, risk = self.assess(request)
        if self.level <= 3:  # no competitor: the first option at its list price always wins
            first = request["options"][0]
            return {first["kind"]: first["listPriceCents"]} if first["listPriceCents"] >= cost else {}
        place = request["legs"][0]["pickup"]["placeId"]
        market, trust, pressure = self.market.multiple(place, self.now), self.trust_boost(request["dealerId"]), self.pressure_factor()
        declined = self.declined(request, scarcity, risk)
        far = max(request["referenceFeeCents"], 100) * DECLINE_MULT
        units: dict[str, float] = {}
        prices: dict[str, int] = {}
        for option in request["options"]:
            kind, listed = option["kind"], max(1, option["listPriceCents"])
            units[kind] = listed * market * OPTION_FACTOR[kind]
            price = max(listed * DECLINE_MULT, far) if kind in declined else max(units[kind] * trust * pressure, cost * OPTION_COST[kind])
            prices[kind] = max(1, int(price + 0.5))
        self.drafts[request["id"]] = Quoted(place, trust, units, prices)
        return prices

    def declined(self, request: PriceRequest, scarcity: float, risk: float) -> set[str]:
        """The options to decline while the delivery minimum is in danger. Same-day work goes first, then the risky cars:
        where drivers are few, or the note adds risk. A decline is a quote far above the list, and it costs nothing."""
        share = self.at_risk_share()
        if share < GUARD_SOFT:
            return set()
        risky = scarcity >= GUARD_SCARCITY or risk >= GUARD_RISK or (share >= GUARD_HARD and scarcity >= GUARD_SCARCITY_HARD)
        return {o["kind"] for o in request["options"] if risky or o["kind"] == "same_day"}

    def assess(self, request: PriceRequest) -> tuple[float, float, float]:
        """What moving the car will probably cost, with the margin: the typical fee, times what the level teaches about fees,
        the scarcity of drivers where the car is, the size of its dealer group's bulk order against the drivers near its yard,
        the risk in its note, and what drivers dislike (RULES.md: private addresses, evenings, Saturdays, charging, trade-ins;
        the typical fee covers only the drive and the trips). Returns (cost, scarcity, note risk)."""
        first = request["legs"][0]
        premium = any(leg["car"]["premium"] for leg in request["legs"])
        scarcity = self.supply.scarcity(first["pickup"]["placeId"], first["delivery"]["placeId"], premium)
        risk = note_risk(read_note(request["note"], request["arrivedAt"]), request)
        cost = (max(0, request["referenceFeeCents"]) * EXPECTED_FEE[self.level] * self.fee_mult * scarcity
                * self.bulk_factor(request) * risk * self.extras(request) * MARGIN * self.ease)
        return cost, scarcity, risk

    def bulk_factor(self, request: PriceRequest) -> float:
        """30 to 40 cars from one yard need 30 to 40 trips from it. The first cars cost no extra, as many as half the drivers
        near the yard (4 to 12); each car after that adds BULK_SLOPE to the cost."""
        batch_id = request["batchId"]
        if not batch_id:
            return 1.0
        near = self.supply.near.get(request["legs"][0]["pickup"]["placeId"], 0)
        free = min(BULK_FREE_MAX, max(BULK_FREE_MIN, BULK_SUPPLY_SHARE * near)) if self.supply.drivers else BULK_FREE_MIN
        size = max(self.batch_wins[batch_id], 0.75 * self.batches[batch_id])
        return 1 + min(BULK_CAP, BULK_SLOPE * max(0.0, size - free))

    def extras(self, request: PriceRequest) -> float:
        """A multiple for the work that drivers dislike. Per car: a private address at either end, a pickup in the evening or on
        Saturday, a charging stop; once: a trade-in."""
        legs = request["legs"]
        extra = EXTRA_TRADE_IN if len(legs) == 2 else 0.0
        for leg in legs:
            pickup, delivery, car = leg["pickup"], leg["delivery"], leg["car"]
            if pickup["privateAddress"] or delivery["privateAddress"]:
                extra += EXTRA_PRIVATE / len(legs)
            start = pickup["window"]["from"]
            if start // DAY % 7 == 5:
                extra += EXTRA_WEEKEND / len(legs)
            elif start % DAY >= EVENING_FROM:
                extra += EXTRA_EVENING / len(legs)
            km = self.supply.distance_km(pickup["placeId"], delivery["placeId"])
            if car["electric"] and car["chargePercent"] is not None and car["chargePercent"] < 30 and km is not None and km > 200:
                extra += EXTRA_CHARGING / len(legs)
        return 1 + extra

    def confirm(self, request_id: str) -> None:
        """The world accepted the quote: from now on a decision of the dealer teaches the market."""
        if request_id in self.drafts:
            self.quoted[request_id] = self.drafts.pop(request_id)

    def learn_outcome(self, request_id: str, won: bool, option: Optional[str]) -> None:
        """A dealer decided. The market learns only from a quote that was priced at its level (not raised for trust, which
        is known, but for trouble, load or cost): a loss at a price that was high on purpose says nothing about the level."""
        quote = self.quoted.pop(request_id, None)
        self.drafts.pop(request_id, None)
        if quote is None or self.level < 4:
            return
        if won:
            unit, price = quote.units.get(option or ""), quote.prices.get(option or "")
            if not unit or not price:
                return
            ratio = price / (unit * quote.trust)
        else:
            ratio = min((quote.prices[k] / (quote.units[k] * quote.trust) for k in quote.prices if quote.units[k] > 0), default=math.inf)
        if ratio <= LEARN_CLEAN:
            self.market.learn(quote.place, self.now, won)

    # --- Orders and gigs ---

    def win(self, order: Order) -> None:
        if order["id"] in self.requests and order["id"] not in self.orders:  # rejoin can know it already
            self.took += 1
            if self.requests[order["id"]]["batchId"]:
                self.batch_wins[self.requests[order["id"]]["batchId"]] += 1
            self.learn_outcome(order["id"], True, order["option"])
            self.orders[order["id"]] = order
            self.post(order, undelivered(order))

    def change(self, order: Order) -> None:
        if order["id"] in self.orders:
            self.orders[order["id"]] = order
            if self.withdraw(order["id"]):
                self.post(order, undelivered(order))

    def repost(self, gig_id: str) -> None:
        """Posts a gig again when one of its windows ended before a driver booked it."""
        action = self.posts.get(gig_id)
        # A gig the desk has already taken off and posted again, after a change of plan, needs nothing.
        if action is None or not any(self.leg_gigs.get(s["orderId"], {}).get(s["leg"]) == gig_id for s in action["stops"]):
            return
        for stop in action["stops"]:
            self.leg_gigs.get(stop["orderId"], {}).pop(stop["leg"], None)
        order = self.orders.get(action["stops"][0]["orderId"])
        if order is None:
            return
        # One of its windows has ended, perhaps at a site's closing time before the window's own end: post without its own stop windows.
        stops = without_windows(action["stops"], float("inf"))
        self.outbox.append({**action, "stops": stops, "schedule": self.fees_for(order, pickups(action), self.current_fee(gig_id), legs_of(action))})

    def revive(self, gig_id: str) -> None:
        """After a drop or a no-show the gig is open again on its old schedule. A driver who gave it back takes it again only
        for more, so the climb starts again from the fee it had reached."""
        action = self.posts.get(gig_id)
        if action is None or not any(self.leg_gigs.get(s["orderId"], {}).get(s["leg"]) == gig_id for s in action["stops"]):
            return
        order = self.orders.get(action["stops"][0]["orderId"])
        if order is not None:
            self.reprice(order, gig_id, action)

    def reprice(self, order: Order, gig_id: str, action: PostAction) -> None:
        schedule = self.fees_for(order, pickups(action), self.current_fee(gig_id) * 1.05, legs_of(action))
        self.schedules[gig_id] = schedule
        self.outbox.append({"type": "reprice", "gigId": gig_id, "schedule": schedule})

    def raise_fees(self, order_id: str) -> None:
        """From the order's deadline warning on, its gigs that no driver holds go at the steep fees, and the nearest drivers
        who fit get an offer."""
        order = self.orders.get(order_id)
        if order is None:
            return
        if order_id not in self.urgent:
            self.worry(1.0)
            self.fee_mult = min(FEE_MAX, self.fee_mult * 1.03)
        self.urgent.add(order_id)
        self.send_post_of(order_id)
        for gig_id in self.gigs_of(order_id):
            if gig_id not in self.booked:
                self.reprice(order, gig_id, self.posts[gig_id])
                self.offer_nearby(order, gig_id)

    def offer_nearby(self, order: Order, gig_id: str) -> None:
        """A direct offer reaches drivers whom a notification does not: up to three of the nearest drivers who fit the gig."""
        action = self.posts[gig_id]
        if not self.supply.drivers or not 7 * HOUR <= self.now % DAY < 23 * HOUR:  # no driver reads an offer from 23:00 to 07:00
            return
        place = order["legs"][action["stops"][0]["leg"]]["pickup"]["placeId"]
        fee = self.schedules[gig_id][0]["feeCents"]
        sent, day, chosen = self.offered.setdefault(gig_id, set()), self.now // DAY, 0
        for driver in self.supply.nearest(place, OFFER_REACH):
            if chosen == 3:
                break
            if driver["id"] in sent or self.offers_by_day[(driver["id"], day)] >= 3:
                continue
            if action["audience"]["experienced"] and not experienced_driver(driver):
                continue
            if action["audience"]["manualLicence"] and driver["licence"] != "manual":
                continue
            sent.add(driver["id"])
            self.offers_by_day[(driver["id"], day)] += 1
            chosen += 1
            self.outbox.append({"type": "offer", "gigId": gig_id, "driverId": driver["id"], "feeCents": fee})

    def current_fee(self, gig_id: str) -> float:
        """The fee in the app now: the latest step of the gig's schedule that has started."""
        started = [step["feeCents"] for step in self.schedules.get(gig_id, []) if step["at"] <= self.now]
        return float(started[-1]) if started else 0.0

    def fee_scarcity(self, order: Order, legs: Optional[list[int]]) -> float:
        """Drivers are few around some pickups, and "a car far from any driver needs a fee that pays for the whole trip": the
        fees start higher there, by FEE_SCARCITY of the scarcity premium that the quote already carries."""
        chosen = [order["legs"][leg] for leg in (legs or [0])]
        first = chosen[0]
        scarcity = self.supply.scarcity(first["pickup"]["placeId"], first["delivery"]["placeId"], any(leg["car"]["premium"] for leg in chosen))
        return 1 + FEE_SCARCITY * (scarcity - 1)

    def fees_for(self, order: Order, cars: int, floor: float = 0.0, legs: Optional[list[int]] = None) -> list[FeeStep]:
        """The fees for some of the order's cars from now: rising, or steep once its deadline warning has come or is due.
        They start at the typical fee, scaled by what bookings have taught and by the scarcity of drivers at the pickup, and
        no step is under `floor`. They stop short of 1.5 times the cars' share of the price, unless the first fee is higher
        already. The cap is higher inside the warning (a late car costs a refund) and higher again once the order is late
        (a car undelivered at the end costs twice its price)."""
        share = cars / len(order["legs"])
        fee = self.requests[order["id"]]["referenceFeeCents"] * share * self.fee_mult * self.fee_scarcity(order, legs)
        urgent = order["id"] in self.urgent or order["lastPickupAt"] - self.now <= URGENT_LEAD
        steps = steep(self.now, fee, self.level) if urgent else rising(self.now, fee, self.level)
        steps = [{"at": s["at"], "feeCents": max(s["feeCents"], int(floor + 0.5))} for s in steps]  # a floor keeps the fees from falling
        cap_share = CAP_LATE if order["deadline"] < self.now else CAP_URGENT if urgent else FEE_CAP_SHARE
        cap = max(steps[0]["feeCents"], int(order["priceCents"] * share * cap_share))
        return [{"at": s["at"], "feeCents": min(s["feeCents"], cap)} for s in steps]

    def retry(self, order_id: str, leg: int, reason: str) -> None:
        if order_id in self.orders:
            order = self.orders[order_id]
            self.leg_gigs.get(order_id, {}).pop(leg, None)  # the world has taken the car out of the failed gig
            car_not_there = reason in ("not_ready", "closed")
            pickup_from = retry_from(order, self.requests[order_id], leg, self.now) if car_not_there else 0
            self.post(order, [leg], Retry(pickup_from, manual_licence=reason == "licence", plates=reason == "no_plates"))

    def post(self, order: Order, legs: list[int], retry: Retry = Retry()) -> None:
        request = self.requests[order["id"]]
        facts = read_note(request["note"], request["arrivedAt"])
        retry = retry._replace(manual_licence=retry.manual_licence or facts.manual, plates=retry.plates or facts.plates)
        # A premium car needs an experienced driver: the fine is €150 a car.
        audience: Audience = {"experienced": any(leg["car"]["premium"] for leg in order["legs"]), "manualLicence": retry.manual_licence}
        stops = without_windows(stops_for(order, legs, request["note"], request["arrivedAt"], retry, self.supply), self.now)
        self.outbox.append({"type": "post", "stops": stops, "audience": audience, "schedule": self.fees_for(order, len(legs), legs=legs)})
        if len(legs) == 1:
            return
        # A trade-in over the 12-hour limit for a gig with two cars goes as one gig per car.
        self.send()
        if not self.gigs_of(order["id"]):
            for leg in legs:
                self.post(order, [leg], retry)

    def gigs_of(self, order_id: str) -> list[str]:
        """The gigs that carry the order's undelivered legs: one, or one for each leg of a split trade-in."""
        return list(dict.fromkeys(self.leg_gigs.get(order_id, {}).values()))

    def carry(self, action: PostAction, gig_id: str) -> None:
        """Remembers a gig that the world accepted: the post that made it, and the legs it carries."""
        for stop in action["stops"]:
            self.leg_gigs.setdefault(stop["orderId"], {})[stop["leg"]] = gig_id
            self.first_posted.setdefault(stop["orderId"], self.now)
        self.posts[gig_id] = action
        self.schedules[gig_id] = action["schedule"]

    def withdraw(self, order_id: str) -> bool:
        """Takes an order's gigs out of the drivers' app. False, and nothing withdrawn, when a driver holds one of them."""
        self.send_post_of(order_id)
        gigs = self.gigs_of(order_id)
        if any(gig_id in self.booked for gig_id in gigs):
            return False
        self.leg_gigs.pop(order_id, None)
        self.outbox += [{"type": "withdraw", "gigId": gig_id} for gig_id in gigs]
        return True

    def send_post_of(self, order_id: str) -> None:
        """A post of the order still in the outbox goes to the world now, so the desk knows its gig. Each gig carries one order."""
        if any(action["type"] == "post" and action["stops"][0]["orderId"] == order_id for action in self.outbox):
            self.send()

    def send(self) -> None:
        """Sends the outbox now. In practice, the outbox goes with the next advance instead."""
        actions, self.outbox = self.outbox, []
        if actions:
            self.read_results(actions, self.world.act(actions))

    def read_results(self, actions: list[Action], results: list[ActionResult]) -> None:
        """Remembers the gig of each post that the world accepted, and the quotes that it took."""
        for action, result in zip(actions, results):
            if result["ok"] is not True:
                continue
            if action["type"] == "post":
                self.carry(action, result["id"])
            elif action["type"] == "quote":
                self.confirm(action["requestId"])


def retry_from(order: Order, request: PriceRequest, leg: int, now: int) -> int:
    """When to try a car again after a visit found it not ready or the site closed: two hours on, or,
    when the dealer moved the deadline for a car that is late, when the car should be ready by the
    same delay."""
    promised = next((o["deadline"] for o in request["options"] if o["kind"] == order["option"]), order["deadline"])
    delay = max(0, order["deadline"] - promised)
    return max(now + 2 * HOUR, order["legs"][leg]["pickup"]["window"]["from"] + delay)


def undelivered(order: Order) -> list[int]:
    return [leg for leg in range(len(order["legs"])) if order["deliveredAt"][leg] is None]


def pickups(action: PostAction) -> int:
    """The cars a post carries."""
    return sum(1 for stop in action["stops"] if stop["kind"] == "pickup")


def legs_of(action: PostAction) -> list[int]:
    """The legs a post carries, in order."""
    return [stop["leg"] for stop in action["stops"] if stop["kind"] == "pickup"]


def without_windows(stops: list[StopRef], before: float) -> list[StopRef]:
    """Leaves out the stop windows that end before `before`, so the world's window applies: once the order's
    window has ended, it runs to the end of the world."""
    return [{k: v for k, v in stop.items() if k != "window"} if "window" in stop and stop["window"]["to"] < before else stop for stop in stops]


def rising(now: int, fee: float, level: int) -> list[FeeStep]:
    """The fee at once, then 10% of it more every hour: six steps, up to 1.5 times. Level 1 takes one fee only."""
    # int(x + 0.5) rounds halves up, as JavaScript's Math.round does, so both starters post the same fees.
    return [{"at": now + k * HOUR, "feeCents": max(1, int(fee * (1 + k / 10) + 0.5))} for k in range(1 if level == 1 else 6)]


def steep(now: int, fee: float, level: int) -> list[FeeStep]:
    """1.5 times the fee at once, then a quarter of it more every half hour, up to 2.5 times. Level 1 takes one fee only."""
    return [{"at": now + k * 30, "feeCents": max(1, int(fee * (1.5 + k / 4) + 0.5))} for k in range(1 if level == 1 else 5)]


def stops_for(order: Order, legs: list[int], note: Optional[Note], arrived_at: int, retry: Retry, supply: Optional[Supply] = None) -> list[StopRef]:
    """A pickup and a delivery for each leg. An electric car under 30% gets a charging stop when its route is over 200 km
    (or may be: without the map, always). A stop has its own window only when the note or the retry makes it narrower.
    Without one, the world plans the pickup in the order's window and the delivery by the deadline."""
    stops: list[StopRef] = []
    for leg in legs:
        car, pickup, delivery = order["legs"][leg]["car"], order["legs"][leg]["pickup"], order["legs"][leg]["delivery"]
        noted = window_from_note(note, pickup["window"], "pickup", arrived_at, pickup["privateAddress"])
        start = max(noted["from"], retry.pickup_from)
        pickup_window: TimeWindow = {"from": start, "to": max(noted["to"], start)}
        delivery_base: TimeWindow = {"from": delivery["window"]["from"], "to": max(delivery["window"]["from"], order["deadline"])}
        delivery_window = window_from_note(note, delivery_base, "delivery", arrived_at, delivery["privateAddress"])
        km = supply.distance_km(pickup["placeId"], delivery["placeId"]) if supply else None
        low_battery = car["electric"] and car["chargePercent"] is not None and car["chargePercent"] < 30 and (km is None or km > 200)
        first: StopRef = {"orderId": order["id"], "leg": leg, "kind": "pickup", "withPlates": retry.plates, "withCharging": low_battery}
        second: StopRef = {"orderId": order["id"], "leg": leg, "kind": "delivery"}
        stops += [narrower(first, pickup_window, pickup["window"]), narrower(second, delivery_window, delivery_base)]
    return stops


def narrower(stop: StopRef, window: TimeWindow, base: TimeWindow) -> StopRef:
    return {**stop, "window": window} if window["from"] > base["from"] or window["to"] < base["to"] else stop


# --- Notes -----------------------------------------------------------------------------------------------------------------
# RULES.md: days count from the day the request arrives; a time without a day is on that day; a weekday name is 2 to 6 days on;
# digits are a 24-hour clock unless am/pm follows; an hour in words is the working hour with that number (9 to 17); "half three" is
# 14:30; a correction counts; a day or time that the note denies does not.
# re.ASCII makes \b and \d mean what they mean in JavaScript, so both starters read a note alike.
_A = re.ASCII
# "Not before", "at the earliest" and a ready time all mean from then on; the keyword nearest to the time decides, "not before" over "before".
AFTER = re.compile(r"jälkeen|alkaen|aikaisintaan|\bafter\b|\bfrom\b|earliest|not before|ei ennen|ready (?:by|at|from)|valmis klo|klar (?:kl\.?|vid)|valmis|\bready\b|vasta\b|efter|tidigast|inte före|\bklar\b|först|\bfrån\b", _A)
BEFORE = re.compile(r"ennen|mennessä|viim|\bbefore\b|\bby\b|latest|före|senast|\buntil\b|\basti\b|saakka|tills\b", _A)
DELIVERY = re.compile(r"toimit|perillä|perille|vastaanot|kohde|deliver|arrive|destination|receiv|leverera|leverans|framme|mottag", _A)
PICKUP = re.compile(r"nouto|noutaa|noudet|hakea|\bhae\b|\bhaku|pick ?-?up|collect|hämta|avhämt|handover|luovut|\brelease|maksu|betal|payment|\bready\b|valmis|\bklar\b|availab|saatavilla|tillgäng", _A)
CLOCK = re.compile(r"(?:klo|kl\.?)\s*(\d{1,2})(?:[.:](\d{2}))?|\b(\d{1,2})[.:](\d{2})\b", _A)
AMPM = re.compile(r"\b(\d{1,2})(?:[.:](\d{2}))?\s*(am|pm)\b", _A)
KEY_HOUR = re.compile(r"\b(?:by|before|after|until|latest|earliest|at|ennen|jälkeen|mennessä|asti|före|efter|senast|tidigast)\s+(\d{1,2})(?![\d.:]|\s*(?:am\b|pm\b|km|%|€|eur|kpl|pcs|st\b|cars?\b|autoa?\b|bilar))", _A)
# Hours in words, in English, Finnish (with the endings that a time takes) and Swedish. "Half three" is half an hour before three.
HOUR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "yksi": 1, "yhden": 1, "yhteen": 1, "kaksi": 2, "kahden": 2, "kahteen": 2, "kahta": 2, "kolme": 3, "kolmen": 3, "kolmeen": 3, "kolmea": 3,
    "neljä": 4, "neljän": 4, "neljään": 4, "neljää": 4, "viisi": 5, "viiden": 5, "viiteen": 5, "viittä": 5, "kuusi": 6, "kuuden": 6, "kuuteen": 6,
    "seitsemän": 7, "kahdeksan": 8, "yhdeksän": 9, "yhdeksään": 9, "kymmenen": 10, "kymmeneen": 10, "yksitoista": 11, "kaksitoista": 12,
    "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5, "sex": 6, "sju": 7, "åtta": 8, "nio": 9, "tio": 10, "elva": 11, "tolv": 12,
}
_WORDS = "|".join(sorted(HOUR_WORDS, key=len, reverse=True))
HALF_CLOCK = re.compile(r"\b(?:half|halv|puoli)\s+(" + _WORDS + r")(?![a-zäöå])", _A)
WORD_CLOCK = re.compile(r"(?:\bklo|\bkl\.?|\bby|\bbefore|\bafter|\bat|\buntil|\bennen|\bmennessä|\bföre|\befter|\bsenast|\bfrom|\btidigast)\s+(" + _WORDS + r")(?![a-zäöå])", _A)
# Opening hours as a range: "9-16", "klo 9.00-14.30", "kl 10-14".
RANGE = re.compile(r"\b(\d{1,2})(?:[.:](\d{2}))?\s*[-–]\s*(\d{1,2})(?:[.:](\d{2}))?\b", _A)
NEGATION = re.compile(r"(?:\bnot|\bei|\binte|\beikä|\bno)\s*(?:siis\s+)?$", _A)
CORRECTION = re.compile(r"actually|correction|i mean|sorry|scratch that|oops|oikeastaan|tarkoitan|anteeksi|oikaisu|ei vaan|snarare|jag menar|rättelse", _A)
# Days: relative words, and weekday names (long forms, and the short forms that cannot be another word).
RELATIVE_DAY = re.compile(r"(ylihuomenna|i övermorgon|övermorgon|day after tomorrow|day after tmrw|day after tmr)|(huomenna|i morgon|imorgon|tomorrow|tmrw|tmr)|(tänään|idag|today)", _A)
WEEKDAY_NAMES = {
    "maanantai": 0, "tiistai": 1, "keskiviikko": 2, "torstai": 3, "perjantai": 4, "lauantai": 5, "sunnuntai": 6,
    "måndag": 0, "tisdag": 1, "onsdag": 2, "torsdag": 3, "fredag": 4, "lördag": 5, "söndag": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}
WEEKDAY_SHORT = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
                 "mån": 0, "tis": 1, "ons": 2, "tors": 3, "fre": 4, "lör": 5, "sön": 6}
WEEKDAY_FI_SHORT = {"ma": 0, "ti": 1, "ke": 2, "to": 3, "pe": 4, "la": 5, "su": 6}  # only with a full stop: "to." is Thursday, "to" is not
WEEKDAY = re.compile(
    r"(?<![a-zäöå])(" + "|".join(sorted(WEEKDAY_NAMES, key=len, reverse=True)) + r")[a-zäöå]*"
    r"|(?<![a-zäöå])(" + "|".join(sorted(WEEKDAY_SHORT, key=len, reverse=True)) + r")(?![a-zäöå])"
    r"|(?<![a-zäöå])(" + "|".join(WEEKDAY_FI_SHORT) + r")\.", _A)
MANUAL = re.compile(r"manuaal|manuell|\bmanual\b|käsivaihde|\bstick\b", _A)
PLATES_WORD = r"(?:plates?|rekisterikilp\w*|kilvet|kilvi|registreringsskyl\w*|skyltar|nummerplåt\w*)"
NO_PLATES = re.compile(r"(?:\bno\b|without|\bei\b|ilman|utan|saknar|missing|puuttu\w*|lacks?)\W+(?:\w+\W+){0,3}?" + PLATES_WORD + r"|" + PLATES_WORD + r"\W+(?:\w+\W+){0,3}?(?:missing|puuttu\w*|saknas|needed|tarvitaan|behövs)", _A)
SUMMER_TYRES = re.compile(r"summer tyres?|summer tires?|kesäreng|kesärenk|sommardäck", _A)
BOUNDARY = re.compile(r"[;,]|(?<!\b\w\w)\.\s|\s(?:and|ja|och)\s", _A)  # a clause ends at a comma, a full stop ("to." is not one), "and"
NEARBY = 24  # characters between a time and the keyword or day that belongs to it
KIND_NEARBY = 40  # characters between a time and the word that says pickup or delivery


class Mention(NamedTuple):
    kind: str  # "after", "before" or "hours"
    stop: str  # "pickup" or "delivery"
    at: int  # the minute, in world time, for after and before; for hours, the opening, in minutes of the day
    close: int = 0  # for hours: the closing, in minutes of the day
    explicit: bool = False  # the note names the day


class Facts(NamedTuple):
    mentions: tuple = ()
    manual: bool = False
    plates: bool = False
    summer_tyres: bool = False


def clock_minutes(hour: int, minute: int) -> Optional[int]:
    return None if hour > 24 or minute > 59 else hour * HOUR + minute


def working_hour(word: str) -> Optional[int]:
    """An hour in words is the working hour with that number, from 9 to 17: "two" is 14:00. 6, 7 and 8 have none."""
    n = HOUR_WORDS[word]
    return n if 9 <= n <= 12 else n + 12 if 1 <= n <= 5 else None


def distance(span: tuple[int, int], hit: re.Match) -> int:
    if hit.end() <= span[0]:
        return span[0] - hit.end()
    return max(0, hit.start() - span[1])


def clause(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """The part of the note between the commas, full stops and "and" around the span: what a time's keywords must be in."""
    lo, hi = 0, len(text)
    for hit in BOUNDARY.finditer(text):
        if hit.end() <= span[0]:
            lo = hit.end()
        elif hit.start() >= span[1]:
            hi = hit.start()
            break
    return lo, hi


def nearest_kind(text: str, span: tuple[int, int], first: re.Pattern, second: re.Pattern, limit: int) -> Optional[int]:
    """0 when a match of `first` is nearest to the span, 1 for `second`, None when neither is within `limit` in the span's clause.
    A tie goes to `first`."""
    lo, hi = clause(text, span)
    best: Optional[tuple[int, int]] = None
    for which, pattern in enumerate((first, second)):
        for hit in pattern.finditer(text):
            d = distance(span, hit)
            if lo <= hit.start() and hit.end() <= hi and d <= limit and (best is None or d < best[0]):
                best = (d, which)
    return None if best is None else best[1]


def day_words(text: str, arrived_at: int) -> list[tuple[int, int]]:
    """(position, days after the arrival day) for each day that the text names."""
    arrival_weekday = arrived_at // DAY % 7
    found: list[tuple[int, int]] = []
    for hit in RELATIVE_DAY.finditer(text):
        found.append((hit.start(), 2 if hit.group(1) else 1 if hit.group(2) else 0))
    for hit in WEEKDAY.finditer(text):
        name = hit.group(1) or hit.group(2) or hit.group(3)
        weekday = WEEKDAY_NAMES.get(name, WEEKDAY_SHORT.get(name, WEEKDAY_FI_SHORT.get(name)))
        if weekday is not None and 2 <= (weekday - arrival_weekday) % 7 <= 6:  # a note never names the arrival day or the next day by its weekday
            found.append((hit.start(), (weekday - arrival_weekday) % 7))
    return found


@lru_cache(maxsize=4096)
def read_text(text: str, arrived_at: int) -> Facts:
    text = text.lower()
    arrival_day = arrived_at // DAY * DAY
    default_kind = "delivery" if DELIVERY.search(text) else "pickup"

    def stop_of(span: tuple[int, int]) -> str:
        nearest = nearest_kind(text, span, DELIVERY, PICKUP, KIND_NEARBY)
        return default_kind if nearest is None else "delivery" if nearest == 0 else "pickup"

    mentions: list[Mention] = []
    # Opening hours hold every day: they move a window's start to the opening and its end to the closing.
    for hit in RANGE.finditer(text):
        opens, closes = clock_minutes(int(hit[1]), int(hit[2] or 0)), clock_minutes(int(hit[3]), int(hit[4] or 0))
        if opens is not None and closes is not None and opens < closes:
            mentions.append(Mention("hours", stop_of(hit.span()), opens, closes))
    masked = RANGE.sub(lambda hit: " " * len(hit.group()), text)  # the same positions, without the ranges
    # Every way to write a time, then the ones that do not overlap an earlier one.
    found: list[tuple[int, int, int, int]] = []  # (start, end, minute of the day, priority)
    for hit in CLOCK.finditer(masked):
        m = clock_minutes(int(hit[1] or hit[3]), int(hit[2] or hit[4] or 0))
        if m is not None:
            found.append((hit.start(), hit.end(), m, 0))
    for hit in AMPM.finditer(masked):
        hour = int(hit[1])
        if 1 <= hour <= 12 and int(hit[2] or 0) <= 59:
            found.append((hit.start(), hit.end(), (hour % 12 + (12 if hit[3] == "pm" else 0)) * HOUR + int(hit[2] or 0), 1))
    for hit in KEY_HOUR.finditer(masked):
        m = clock_minutes(int(hit[1]), 0)
        if m is not None:
            found.append((hit.start(), hit.end(), m, 2))
    for hit in HALF_CLOCK.finditer(masked):
        hour = working_hour(hit[1])
        if hour is not None:
            found.append((hit.start(), hit.end(), hour * HOUR - 30, 3))
    for hit in WORD_CLOCK.finditer(masked):
        hour = working_hour(hit[1])
        if hour is not None:
            found.append((hit.start(), hit.end(), hour * HOUR, 4))
    days = day_words(text, arrived_at)
    timed: list[Mention] = []
    last_end = 0
    for start, end, minute, _ in sorted(found, key=lambda f: (f[0], f[3])):
        if start < last_end:
            continue
        last_end = end
        if NEGATION.search(text[max(0, start - 12):start]):  # "not 3pm", "ei klo 15": a time that the note denies
            continue
        which = nearest_kind(text, (start, end), AFTER, BEFORE, NEARBY)
        if which is None:
            continue
        near_days = [(abs(p - start), d) for p, d in days if abs(p - start) <= NEARBY]
        offset = min(near_days)[1] if near_days else 0
        timed.append(Mention("after" if which == 0 else "before", stop_of((start, end)), arrival_day + offset * DAY + minute, 0, bool(near_days)))
    if CORRECTION.search(text):  # the correction counts: the last time for each kind and stop, on the day of the one that it corrects
        last: dict[tuple[str, str], Mention] = {}
        for m in timed:
            before = last.get((m.kind, m.stop))
            if before is not None and before.explicit and not m.explicit:
                m = m._replace(at=before.at // DAY * DAY + m.at % DAY, explicit=True)
            last[(m.kind, m.stop)] = m
        timed = list(last.values())
    mentions += timed
    manual_hit = MANUAL.search(text)
    manual = manual_hit is not None and not NEGATION.search(text[max(0, manual_hit.start() - 12):manual_hit.start()])  # "ei manuaali" is not manual
    return Facts(tuple(mentions), manual, bool(NO_PLATES.search(text)), bool(SUMMER_TYRES.search(text)))


def read_note(note: Optional[Note], arrived_at: int) -> Facts:
    return read_text(note["text"], arrived_at) if note and note.get("text") else Facts()


def note_risk(facts: Facts, request: PriceRequest) -> float:
    """A multiple for the cost of a car whose note adds risk."""
    risk = 1.0
    if facts.manual:
        risk += 0.10  # fewer drivers may take it
    if facts.plates:
        risk += 0.05  # 30 minutes more
    if facts.summer_tyres:
        risk += 0.15  # snow can stop it
    last = max(o["deadline"] for o in request["options"])
    if any(m.kind == "before" and m.stop == "delivery" and m.at < last - 30 for m in facts.mentions):
        risk += 0.10  # a tighter deadline than the option's
    return risk


def window_from_note(note: Optional[Note], window: TimeWindow, stop: str, arrived_at: int, private: bool = False) -> TimeWindow:
    """Narrows a stop's window by what the note says: "nouto ennen 15.30", "klo 12 jälkeen", "deliver by tomorrow 19:00",
    "efter kl. 12", "klar först på onsdag kl 12", "not before 16:00", "ready by two", and opening hours as a range,
    "arkisin 9-16". A time with no day is on the day the request arrived, as RULES.md says; when that is before the
    window starts, it is taken on the window's own day, because it is then most likely a closing time that holds every day.
    A car that is ready later than its window ends can still be collected: its stop window then starts at the ready time."""
    facts = read_note(note, arrived_at)
    start, end, ready_later = window["from"], window["to"], False
    day = window["from"] // DAY * DAY
    for m in facts.mentions:
        if m.stop != stop:
            continue
        if m.kind == "hours":
            start, end = max(start, start // DAY * DAY + m.at), min(end, end // DAY * DAY + m.close)
        elif m.kind == "after":
            start = max(start, m.at)
            ready_later = ready_later or start > window["to"]
        else:
            at = m.at if m.explicit or m.at >= window["from"] else day + m.at % DAY
            end = min(end, at)
    if start <= end:
        return {"from": start, "to": end}
    # A contradiction: keep the window. But a ready time after the window's end moves the stop on, except at a private address.
    return {"from": start, "to": start} if ready_later and not private and end >= window["from"] else window


# --- Playing ---------------------------------------------------------------------------------------------------------------

def play_practice(world: World, level: int) -> None:
    """Steps the world from one event that needs an answer to the next, and stops at each midnight for a report."""
    snapshot = world.snapshot()
    desk, ends_at = Desk(world, level, snapshot["now"], load_supply(world)), snapshot["endsAt"]
    while desk.now < ends_at:
        day, midnight = desk.now // DAY, (desk.now // DAY + 1) * DAY
        # One request a step: the advance applies the outbox, then moves the clock.
        actions, desk.outbox = desk.outbox, []
        step = world.advance(until=WAKE_ON, max=min(ends_at, midnight), actions=actions)
        desk.read_results(actions, step["results"])
        desk.now = step["now"]
        for event in step["events"]:
            desk.handle(event)
        if desk.now // DAY > day:
            report(WEEKDAYS[day % 7], world.score())


def play_final(world: World) -> None:
    """Rebuilds the books, then answers events as they come. After a second without an event,
    it reads the clock."""
    history = world.events()
    snapshot = world.snapshot()
    if snapshot["now"] >= snapshot["endsAt"]:
        return
    level = next((w["level"] for w in world.client.worlds() if w["worldId"] == world.id), 8)
    desk = Desk(world, level, snapshot["now"], load_supply(world))
    desk.rejoin(snapshot, history)
    desk.send()
    for event in world.stream(after=history[-1]["seq"] if history else -1, idle=1.0):
        day = desk.now // DAY
        desk.now = world.client.final()["now"] if event is None else max(desk.now, event["at"])
        if event is not None:
            desk.handle(event)
        desk.send()
        if desk.now // DAY > day:
            report(WEEKDAYS[day % 7], world.score())
        if desk.now >= snapshot["endsAt"]:
            return


def report(label: str, score: Score) -> None:
    orders = f"{score['wonOrders']:>4} orders  {score['onTimeShare']:6.1%} on time  {score['undeliveredShare']:5.1%} not delivered"
    print(f"{label:<9}  margin €{score['marginCents'] / 100:>11,.2f}  revenue €{score['revenueCents'] / 100:>11,.2f}  {orders}", flush=True)


def next_final(client: Client) -> World:
    """Your copy of the final that is set or runs now. Waits while none is set and while the last one has
    ended: the organisers set a test final 15 minutes before its start, and the final 30 minutes before."""
    said = False
    while True:
        try:
            world: Optional[World] = client.world(client.final()["worldId"])
        except SpreadError as error:
            if error.status != 404:
                raise
            world = None
        if world is not None:
            snapshot = world.snapshot()
            if snapshot["now"] < snapshot["endsAt"]:
                return world
        if not said:
            print("No final is set. Waiting for it: the organisers set a test final 15 minutes before its start, and the final 30 minutes before.", flush=True)
            said = True
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plays a Spread world. Set SPREAD_URL and SPREAD_KEY first.")
    parser.add_argument("--level", type=int, choices=range(1, 9), default=8, help="the practice level, 1 to 8 (default 8)")
    parser.add_argument("--seed", help="any text; the same seed gives the same world")
    parser.add_argument("--board", action="store_true", help="play your next ranking run at the level; it counts in the practice rankings")
    parser.add_argument("--final", action="store_true", help="play your copy of the final from the live stream")
    args = parser.parse_args()
    try:
        client = Client()
        score = None
        if args.final:
            world = None
            # A server restart drops every connection. The desk then rebuilds its books from the server, as after its own restart.
            while True:
                try:
                    if world is None:
                        world = next_final(client)
                        print(f"Final world {world.id}.", flush=True)
                    play_final(world)
                    score = world.score()
                    break
                except Unreachable as error:
                    print(f"spread: {error}. Trying again in 5 seconds.", file=sys.stderr, flush=True)
                    time.sleep(5)
        else:
            world = client.create(level=args.level, seed=args.seed, board=args.board)
            # A ranking run that a team deletes plays to its end without the team's system, and counts.
            name = f"{'Ranking run' if args.board else 'Practice world'} {world.id}"
            stops = "Ctrl-C or an error stops your system: the run then plays to its end without it, and counts." if args.board else "Ctrl-C or an error stops the run and deletes the world."
            print(f"{name}, level {args.level}. {stops}", flush=True)
            try:
                play_practice(world, args.level)
            except BaseException as error:  # a world left behind holds one of the team's eight practice slots
                try:
                    world.delete()
                    deleted = True
                except SpreadError:
                    deleted = False
                after = (f"{name} plays to its end without your system, and counts." if args.board
                         else f"{name} is deleted." if deleted else f"{name} is not deleted: it holds one of your eight practice slots.")
                if isinstance(error, KeyboardInterrupt):
                    sys.exit(f"Stopped. {after}")
                print(f"Stopped by the error below. {after}", file=sys.stderr)
                raise
        score = score or world.score()
        report("End", score)
        print("Ranked." if score["ranked"] else "Not ranked: no order won." if score["wonOrders"] == 0 else "Not ranked: the delivery minimum is 90% of the orders on time.")
    except SpreadError as error:
        sys.exit(f"spread: {error}")
    except KeyboardInterrupt:
        sys.exit("Stopped. The final world keeps running without your system; start it again with --final.")


if __name__ == "__main__":
    main()
