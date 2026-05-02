from __future__ import annotations

import copy
import sys
import time
from typing import Any

try:
    from .ipc import NdjsonClient
except ImportError:
    from ipc import NdjsonClient


class DD2Env:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout: float = 8.0,
        reset_timeout: float = 60.0,
        action_timeout: float = 8.0,
        max_no_ack_retries: int = 3,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self._log(f"connecting to {host}:{port}")
        self.client = NdjsonClient(host=host, port=port, timeout=timeout)
        self.client.connect()
        self._log("connected")
        self.reset_timeout = reset_timeout
        self.action_timeout = action_timeout
        self.max_no_ack_retries = max(1, int(max_no_ack_retries))
        self.last_state: dict[str, Any] | None = None
        self._request_id = 1000
        self._handshake()
        if self.last_state:
            self._log("initial state received")
        else:
            self._log("no initial state yet")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DD2Env] {msg}", file=sys.stderr, flush=True)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _recv_all(self, timeout: float = 0.5, max_wait: float = 2.0) -> list[dict[str, Any]]:
        all_msgs: list[dict[str, Any]] = []
        deadline = time.time() + max_wait
        while time.time() < deadline:
            batch = self.client.recv_lines(timeout=timeout, wait_cycles=1)
            if not batch:
                break
            all_msgs.extend(batch)
        return all_msgs

    @staticmethod
    def _is_live_state(msg: dict[str, Any]) -> bool:
        heroes = msg.get("heroes") or []
        enemies = msg.get("enemies") or []
        return bool(heroes and enemies)

    def _handshake(self) -> None:
        rid = self._next_request_id()
        self.client.send({"type": "ping", "request_id": rid})
        for msg in self._recv_all(timeout=0.4, max_wait=2.0):
            if msg.get("type") == "state":
                self.last_state = msg

    def _state_to_obs(self, state: dict[str, Any]) -> dict[str, Any]:
        active_unit = state.get("active_unit") or {}
        return {
            "in_battle": bool(state.get("in_battle", False)),
            "round": int(state.get("round", 0)),
            "active_side": str(active_unit.get("side", "none")),
            "active_index": int(active_unit.get("index", -1)),
            "heroes": state.get("heroes") or [],
            "enemies": state.get("enemies") or [],
            "legal_actions": state.get("legal_actions") or [],
            "done": bool(state.get("done", False)),
            "heroes_won": state.get("heroes_won"),
        }

    def reset(self) -> dict[str, Any]:
        if self.last_state and self._is_live_state(self.last_state):
            return self._state_to_obs(self.last_state)

        print("Waiting for next battle (start one in-game)...", flush=True)
        deadline = time.time() + self.reset_timeout
        while time.time() < deadline:
            rid = self._next_request_id()
            self.client.send({"type": "ping", "request_id": rid})
            for msg in self._recv_all(timeout=0.4, max_wait=1.0):
                if msg.get("type") == "state":
                    self.last_state = msg
                if self._is_live_state(msg):
                    return self._state_to_obs(msg)
            time.sleep(0.25)
        raise TimeoutError("reset_timeout: no live state received")

    def refresh(self, max_wait: float = 1.0, timeout: float = 0.25) -> dict[str, Any]:
        """Poll the plugin for the latest state without submitting an action."""
        rid = self._next_request_id()
        self.client.send({"type": "ping", "request_id": rid})
        for msg in self._recv_all(timeout=timeout, max_wait=max_wait):
            if msg.get("type") == "state" and self._is_live_state(msg):
                self.last_state = msg
            elif msg.get("type") == "battle_end" and self.last_state is not None:
                reported_won = bool(msg.get("heroes_won", False))
                prev_state = copy.deepcopy(self.last_state)
                won, _ = self._infer_terminal_won_after_action(reported_won, prev_state, action=None, terminal_event=True)
                self.last_state["done"] = True
                self.last_state["heroes_won"] = won
        if self.last_state is None:
            raise RuntimeError("No state received yet")
        return self._state_to_obs(self.last_state)

    def wait_for_terminal(self, max_wait: float = 8.0, poll_interval: float = 0.5) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Poll for a battle_end event or infer a terminal result from live counts."""
        deadline = time.time() + max(0.0, max_wait)
        info: dict[str, Any] = {"reason": "terminal_not_observed"}
        while time.time() < deadline:
            obs = self.refresh(max_wait=max(0.25, poll_interval), timeout=0.25)
            if obs.get("done"):
                won, terminal_info = self._infer_terminal_won_after_action(
                    obs.get("heroes_won"),
                    self.last_state or {},
                    action=None,
                    terminal_event=True,
                )
                info = {"heroes_won": won, "method": "terminal_refresh", **terminal_info}
                return obs, True, info

            won, terminal_info = self._infer_terminal_won(None)
            if won is not None and terminal_info.get("terminal_source") == "alive_counts":
                if self.last_state is not None:
                    self.last_state["done"] = True
                    self.last_state["heroes_won"] = won
                    obs = self._state_to_obs(self.last_state)
                info = {"heroes_won": won, "method": "terminal_inferred", **terminal_info}
                return obs, True, info
            info = terminal_info
            time.sleep(max(0.05, poll_interval))

        if self.last_state is None:
            raise RuntimeError("No state received yet")
        return self._state_to_obs(self.last_state), False, info

    def wait_for_hero_turn(
        self,
        max_wait: float = 60.0,
        poll_interval: float = 0.5,
        remnant_pass_after: float = 4.0,
    ) -> dict[str, Any]:
        """Wait until the live battle is ready for a hero action."""
        if self.last_state is None:
            return self.reset()

        deadline = time.time() + max_wait
        last_report = 0.0
        remnant_seen_at: float | None = None
        remnant_key: tuple[int, int] | None = None
        remnant_passed: set[tuple[int, int]] = set()
        obs = self._state_to_obs(self.last_state)
        while time.time() < deadline:
            if obs.get("done") or obs.get("active_side") == "heroes":
                return obs
            now = time.time()
            active_remnant = self._active_enemy_is_remnant(self.last_state)
            if active_remnant is not None:
                key = (int(obs.get("round", 0)), int(obs.get("active_index", -1)))
                if key != remnant_key:
                    remnant_key = key
                    remnant_seen_at = now
                elif (
                    remnant_seen_at is not None
                    and now - remnant_seen_at >= remnant_pass_after
                    and key not in remnant_passed
                ):
                    self._log(
                        "passing stuck enemy remnant turn "
                        f"round={key[0]} active_index={key[1]} name={active_remnant.get('name')}"
                    )
                    remnant_passed.add(key)
                    obs = self._pass_stuck_remnant_turn()
                    if obs.get("done") or obs.get("active_side") == "heroes":
                        return obs
                    remnant_seen_at = time.time()
                    continue
            else:
                remnant_seen_at = None
                remnant_key = None

            if self.verbose and now - last_report >= 5.0:
                self._log(
                    "waiting for hero turn "
                    f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}"
                )
                last_report = now
            time.sleep(max(0.05, poll_interval))
            obs = self.refresh(max_wait=max(0.25, poll_interval), timeout=0.25)
        raise TimeoutError(
            "enemy_turn_wait timeout: "
            f"active_side={obs.get('active_side')} active_index={obs.get('active_index')}"
        )

    def _active_enemy_is_remnant(self, state: dict[str, Any] | None) -> dict[str, Any] | None:
        if not state:
            return None
        active = state.get("active_unit") or {}
        if str(active.get("side", "none")) != "enemies":
            return None
        active_index = int(active.get("index", -1))
        for idx, enemy in enumerate(state.get("enemies") or []):
            slot = int(enemy.get("slot", idx))
            if slot == active_index and self._is_enemy_remnant(enemy):
                return enemy
        return None

    def _pass_stuck_remnant_turn(self) -> dict[str, Any]:
        if self.last_state is None:
            raise RuntimeError("No state available for remnant pass recovery.")

        prev_state = copy.deepcopy(self.last_state)
        request_id = self._next_request_id()
        self.client.send({"type": "action", "request_id": request_id, "pass_turn": True})

        deadline = time.time() + self.action_timeout
        ack: dict[str, Any] | None = None
        while time.time() < deadline:
            for msg in self._recv_all(timeout=0.25, max_wait=0.5):
                mt = msg.get("type")
                if mt == "ack" and self._msg_request_id(msg) == request_id:
                    ack = msg
                    if not bool(msg.get("ok", False)):
                        self._log(f"stuck remnant pass ack failed: {msg}")
                elif mt == "state" and self._is_live_state(msg):
                    self.last_state = msg
                    if self._state_delta(prev_state, msg):
                        return self._state_to_obs(msg)
                elif mt == "battle_end":
                    reported_won = bool(msg.get("heroes_won", False))
                    won, _ = self._infer_terminal_won_after_action(reported_won, prev_state, action=None, terminal_event=True)
                    self.last_state["done"] = True
                    self.last_state["heroes_won"] = won
                    return self._state_to_obs(self.last_state)
            if ack is None:
                ping_id = self._next_request_id()
                self.client.send({"type": "ping", "request_id": ping_id})

        self._log("stuck remnant pass did not advance state before timeout")
        return self._state_to_obs(self.last_state)

    def _validate_action(self, action: dict[str, Any]) -> None:
        is_pass = bool(action.get("pass_turn"))
        has_skill = all(k in action for k in ("hero_slot", "skill_idx", "target_idx"))
        has_item = ("item_id" in action) and ("target_idx" in action)
        has_move = ("hero_slot" in action) and ("move_delta" in action)
        if sum((1 if is_pass else 0, 1 if has_skill else 0, 1 if has_item else 0, 1 if has_move else 0)) != 1:
            raise ValueError("Action must be exactly one of: pass_turn | skill tuple | item tuple | move tuple")
        target_team = action.get("target_team")
        if target_team is not None and target_team not in ("heroes", "enemies"):
            raise ValueError("target_team must be 'heroes' or 'enemies' if provided")

    @staticmethod
    def _msg_request_id(msg: dict[str, Any]) -> int:
        raw = msg.get("request_id", msg.get("requestId", -1))
        try:
            return int(raw)
        except Exception:
            return -1

    @staticmethod
    def _index_hp(units: list[dict[str, Any]]) -> dict[int, int]:
        return {int(u.get("slot", -1)): int(u.get("hp", 0)) for u in units}

    @staticmethod
    def _is_enemy_remnant(unit: dict[str, Any]) -> bool:
        text = " ".join(
            str(unit.get(key, ""))
            for key in ("id", "name", "archetype_id", "display_name")
        ).lower()
        return any(
            marker in text
            for marker in (
                "corpse",
                "cadaver",
                "remnant",
                "tomb",
                "grave",
                "gravestone",
                "headstone",
                "труп",
                "надгроб",
            )
        )

    def _index_combat_hp(self, units: list[dict[str, Any]], *, side: str) -> dict[int, int]:
        if side != "enemies":
            return self._index_hp(units)
        return {
            int(u.get("slot", -1)): int(u.get("hp", 0))
            for u in units
            if not self._is_enemy_remnant(u)
        }

    def _alive_count(self, units: list[dict[str, Any]], *, side: str = "heroes") -> int:
        alive = 0
        for unit in units:
            if side == "enemies" and self._is_enemy_remnant(unit):
                continue
            hp = int(unit.get("hp", 0))
            if bool(unit.get("alive", hp > 0)):
                alive += 1
        return alive

    def _infer_terminal_won(self, reported_won: bool | None, *, terminal_event: bool = False) -> tuple[bool | None, dict[str, Any]]:
        state = self.last_state or {}
        heroes = state.get("heroes") or []
        enemies = state.get("enemies") or []
        heroes_alive = self._alive_count(heroes, side="heroes")
        enemies_alive = self._alive_count(enemies, side="enemies")
        state_won = state.get("heroes_won")

        inferred: bool | None = None
        source = "alive_counts_pending"
        if terminal_event:
            inferred = heroes_alive > 0
            source = "terminal_heroes_alive" if inferred else "terminal_no_heroes_alive"
        elif enemies_alive == 0 and heroes_alive > 0:
            inferred = True
            source = "alive_counts"
        elif heroes_alive == 0:
            inferred = False
            source = "alive_counts"

        return inferred, {
            "reported_heroes_won": reported_won,
            "state_heroes_won": state_won,
            "terminal_source": source,
            "heroes_alive": heroes_alive,
            "enemies_alive": enemies_alive,
            "heroes_hp": self._index_hp(heroes),
            "enemies_hp": self._index_combat_hp(enemies, side="enemies"),
            "raw_enemies_hp": self._index_hp(enemies),
        }

    def _infer_terminal_won_after_action(
        self,
        reported_won: bool | None,
        prev_state: dict[str, Any],
        action: dict[str, Any] | None,
        *,
        terminal_event: bool = False,
    ) -> tuple[bool | None, dict[str, Any]]:
        inferred, info = self._infer_terminal_won(reported_won, terminal_event=terminal_event)
        state = self.last_state or {}
        has_live_units = bool((state.get("heroes") or []) or (state.get("enemies") or []))
        prev_heroes_alive = self._alive_count(prev_state.get("heroes") or [], side="heroes")
        prev_enemies_alive = self._alive_count(prev_state.get("enemies") or [], side="enemies")
        prev_active_side = str((prev_state.get("active_unit") or {}).get("side", "none"))

        # DD2 may clear combat teams before the post-terminal state can be read.
        # If the final state is empty but heroes were alive in the last combat
        # snapshot, prefer a likely hero win. This covers both direct hero kills
        # and enemy-turn DoT deaths (poison/bleed/burn) before the next hero turn.
        if (
            not has_live_units
            and prev_heroes_alive > 0
            and prev_enemies_alive > 0
        ):
            is_hero_action = action is not None and not action.get("pass_turn") and "skill_idx" in action
            if is_hero_action or prev_active_side == "enemies":
                inferred = True
                info["terminal_source"] = (
                    "hero_action_empty_terminal_state"
                    if is_hero_action
                    else "enemy_turn_empty_terminal_state"
                )
            info["prev_heroes_alive"] = prev_heroes_alive
            info["prev_enemies_alive"] = prev_enemies_alive
            info["prev_active_side"] = prev_active_side
            info["prev_heroes_hp"] = self._index_hp(prev_state.get("heroes") or [])
            info["prev_enemies_hp"] = self._index_combat_hp(prev_state.get("enemies") or [], side="enemies")
            info["prev_raw_enemies_hp"] = self._index_hp(prev_state.get("enemies") or [])
        return inferred, info

    def _state_delta(self, prev: dict[str, Any], nxt: dict[str, Any]) -> bool:
        prev_active = (prev.get("active_unit") or {}).get("index", -1), (prev.get("active_unit") or {}).get("side", "none")
        next_active = (nxt.get("active_unit") or {}).get("index", -1), (nxt.get("active_unit") or {}).get("side", "none")
        if prev_active != next_active:
            return True
        if int(prev.get("round", 0)) != int(nxt.get("round", 0)):
            return True
        if self._index_hp(prev.get("heroes") or []) != self._index_hp(nxt.get("heroes") or []):
            return True
        if self._index_hp(prev.get("enemies") or []) != self._index_hp(nxt.get("enemies") or []):
            return True
        return False

    def _reward(self, prev: dict[str, Any], nxt: dict[str, Any], terminal_won: bool | None) -> float:
        reward = -0.01
        prev_e = self._index_hp(prev.get("enemies") or [])
        next_e = self._index_hp(nxt.get("enemies") or [])
        prev_h = self._index_hp(prev.get("heroes") or [])
        next_h = self._index_hp(nxt.get("heroes") or [])
        for s, hp in prev_e.items():
            reward += 0.1 * max(0, hp - next_e.get(s, hp))
        for s, hp in prev_h.items():
            reward -= 0.1 * max(0, hp - next_h.get(s, hp))
        if terminal_won is True:
            reward += 20.0
        elif terminal_won is False:
            reward -= 20.0
        return reward

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._validate_action(action)
        if self.last_state is None:
            raise RuntimeError("Call reset() before step().")

        prev_state = copy.deepcopy(self.last_state)
        request_id = self._next_request_id()
        self.client.send({"type": "action", "request_id": request_id, **action})

        ack: dict[str, Any] | None = None
        state_after: dict[str, Any] | None = None
        has_delta = False
        deadline = time.time() + self.action_timeout
        no_ack_retries = 0
        while time.time() < deadline:
            got_any = False
            for msg in self._recv_all(timeout=0.25, max_wait=0.5):
                got_any = True
                mt = msg.get("type")
                if mt == "ack" and self._msg_request_id(msg) == request_id:
                    ack = msg
                elif mt == "state":
                    self.last_state = msg
                    state_after = msg
                    if ack is not None and self._state_delta(prev_state, msg):
                        has_delta = True
                        break
                elif mt == "battle_end":
                    reported_won = bool(msg.get("heroes_won", False))
                    won, terminal_info = self._infer_terminal_won_after_action(
                        reported_won,
                        prev_state,
                        action,
                        terminal_event=True,
                    )
                    if self.last_state is not None:
                        self.last_state["done"] = True
                        self.last_state["heroes_won"] = won
                    obs = self._state_to_obs(self.last_state)
                    return (
                        obs,
                        self._reward(prev_state, self.last_state, won),
                        True,
                        {"heroes_won": won, "method": "battle_end", **terminal_info},
                    )
            if has_delta:
                break
            if ack is None:
                if not got_any:
                    no_ack_retries += 1
                ping_id = self._next_request_id()
                self.client.send({"type": "ping", "request_id": ping_id})
                if no_ack_retries >= self.max_no_ack_retries:
                    break

        if ack is None:
            return self._state_to_obs(self.last_state), -0.5, False, {"reason": "no_ack"}
        if not bool(ack.get("ok", False)):
            return self._state_to_obs(self.last_state), -0.2, False, {"reason": ack.get("reason", "ack_failed")}

        # No live state change observed yet: force one last refresh and re-check.
        if not has_delta:
            ping_id = self._next_request_id()
            self.client.send({"type": "ping", "request_id": ping_id})
            for msg in self._recv_all(timeout=0.25, max_wait=1.2):
                if msg.get("type") == "state":
                    self.last_state = msg
                    state_after = msg
                    if self._state_delta(prev_state, msg):
                        has_delta = True
                        break

        if state_after is None:
            return self._state_to_obs(self.last_state), -0.01, False, {"method": ack.get("method"), "reason": "no_post_state"}

        reason = None if has_delta else "no_state_delta_after_ack"
        reward = self._reward(prev_state, state_after, None)
        return self._state_to_obs(state_after), reward, False, {"method": ack.get("method"), "reason": reason}

    def close(self) -> None:
        self.client.close()
