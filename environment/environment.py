"""
CBIC Customs Environment — Per-Task Episode Architecture.
Each task is a fully independent episode (reset → N steps → done=True).
"""

from __future__ import annotations
import random
import uuid
from typing import Optional, Dict, Any
import re

from environment.models import (
    CargoManifest, CaseMetadata, CustomsCase,
    ResetResponse, StepResponse, EnvironmentState,
    Channel, CustomsAction, CustomsState,
    AnomalyType, ANOMALY_SEVERITY,
)
from environment.cases import CASES, CASES_BY_ID, CASES_BY_DIFFICULTY
from environment.graders import (
    AnomalyDetectionGrader,
    ChannelAssignmentGrader,
    SCNGrader,
)

try:
    from openenv.core.env_server import Environment
except ImportError:
    try:
        from openenv_core.env_server import Environment
    except ImportError:
        class Environment:
            pass

# Task name constants
TASK_ANOMALY = "manifest-anomaly-detection"
TASK_CHANNEL = "channel-assignment"
TASK_SCN = "show-cause-notice"

VALID_TASKS = {TASK_ANOMALY, TASK_CHANNEL, TASK_SCN}

# Steps per task
TASK_STEPS = {
    TASK_ANOMALY: 1,
    TASK_CHANNEL: 2,
    TASK_SCN: 3,
}

EXPECTED_ACTIONS = {
    TASK_ANOMALY: ["detect_anomalies"],
    TASK_CHANNEL: ["detect_anomalies", "assign_channel"],
    TASK_SCN: ["detect_anomalies", "assign_channel", "draft_scn"],
}

ANOMALY_LEGAL_BASIS = {
    AnomalyType.REPEAT_VIOLATOR.value: "Section 127, Customs Act 1962",
    AnomalyType.HIGH_RISK_ORIGIN.value: "FATF/OFAC risk advisories",
    AnomalyType.WEIGHT_VOLUME_MISMATCH.value: "WCO risk controls",
    AnomalyType.SEVERE_UNDERVALUATION.value: "Section 14, Customs Act + Valuation Rules 2007",
    AnomalyType.NEW_IEC_HIGH_VALUE.value: "DGFT IEC risk heuristic",
    AnomalyType.SUSPICIOUS_ROUTING.value: "CBIC origin/routing circulars",
    AnomalyType.UNDISCLOSED_RELATED_PARTY.value: "Valuation Rules 2007 (related-party disclosure)",
    AnomalyType.HS_CODE_RISK.value: "HSN classification compliance guidance",
}


def _deterministic_shift_label(boe_number: str) -> str:
    shifts = ["morning", "afternoon", "night"]
    return shifts[sum(ord(c) for c in boe_number) % len(shifts)]


def _queue_pressure(difficulty: str) -> str:
    return {
        "clean": "low",
        "easy": "moderate",
        "medium": "high",
        "hard": "critical",
    }.get(difficulty, "moderate")


class CustomsEnvironment(Environment):
    """
    Reinforcement Learning environment for CBIC customs inspection.
    Supports 3 tasks as independent episodes.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._grader_anomaly = AnomalyDetectionGrader()
        self._grader_channel = ChannelAssignmentGrader()
        self._grader_scn = SCNGrader()
        self._episode_state: Dict[str, Any] = {}
        self._is_active = False

    def get_last_explain(self) -> Dict[str, Any]:
        """Return latest step explanation payload for judge-facing transparency."""
        if not self._episode_state:
            return {"status": "no_episode"}
        return {
            "episode_id": self._episode_state.get("episode_id"),
            "task_name": self._episode_state.get("task_name"),
            "step": self._episode_state.get("current_step", 0),
            "last_step_explain": self._episode_state.get("last_step_explain", {}),
            "trace_id": self._episode_state.get("trace_id"),
        }

    def is_episode_active(self) -> bool:
        """Returns True if an episode is currently in progress."""
        return self._is_active and not self._episode_state.get("done", True)

    def reset(
        self,
        task_name: str = TASK_ANOMALY,
        case_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> ResetResponse:
        """Start a new episode for the given task."""
        if task_name not in VALID_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid: {list(VALID_TASKS)}"
            )

        # Select case
        if case_id and case_id in CASES_BY_ID:
            case: CustomsCase = CASES_BY_ID[case_id]
        elif difficulty and difficulty in CASES_BY_DIFFICULTY and CASES_BY_DIFFICULTY[difficulty]:
            case = self._rng.choice(CASES_BY_DIFFICULTY[difficulty])
        else:
            case = self._rng.choice(CASES)

        max_steps = TASK_STEPS[task_name]
        episode_id = f"EP-{uuid.uuid4().hex[:8].upper()}"

        self._episode_state = {
            "episode_id": episode_id,
            "trace_id": f"TR-{uuid.uuid4().hex[:12].upper()}",
            "task_name": task_name,
            "max_steps": max_steps,
            "current_step": 0,
            "done": False,
            "manifest": case.manifest,
            "metadata": case.metadata,
            "agent_anomalies": [],       # populated after step 1
            # Fix #6: stored for future cross-step validation; not used in SCNGrader
            "agent_channel": None,
            "step_rewards": [],
            "cumulative_reward": 0.0,
            "decision_log": [],
            "last_step_explain": {},
        }
        self._is_active = True

        return ResetResponse(
            episode_id=episode_id,
            task_name=task_name,
            manifest=case.manifest,
            step=0,
            max_steps=max_steps,
        )

    def step(self, action: Dict[str, Any] | CustomsAction) -> StepResponse:
        """Process one agent action and return reward + feedback."""
        if not self._is_active or self._episode_state.get("done"):
            raise RuntimeError("No active episode. Call reset() first.")

        if isinstance(action, CustomsAction):
            action = action.model_dump(exclude_none=True)

        state = self._episode_state
        state["current_step"] += 1
        step_num = state["current_step"]
        task = state["task_name"]
        manifest: CargoManifest = state["manifest"]
        metadata: CaseMetadata = state["metadata"]

        reward = 0.0
        feedback = ""
        details: Dict[str, Any] = {}
        shaping_bonus = 0.0
        shaping_penalty = 0.0
        borderline_penalty = 0.0
        validity_penalty = 0.0
        validity_issues: list[str] = []

        declared_task = action.get("task", "")
        expected_task = EXPECTED_ACTIONS.get(task, [""])[step_num - 1]
        if declared_task != expected_task:
            shaping_penalty += 0.10

        base_details: Dict[str, Any] = {
            "officer_context": {
            "port": manifest.port_of_entry,
            "shift": _deterministic_shift_label(manifest.boe_number),
            "queue_pressure": _queue_pressure(metadata.difficulty),
            "case_id": metadata.case_id,
            }
        }

        # ---------------------------------------------------------------
        # Route action to correct grader
        # ---------------------------------------------------------------
        if step_num == 1:
            # All tasks: detect_anomalies
            predicted = action.get("anomalies", [])
            if not isinstance(predicted, list):
                predicted = []
                validity_penalty += 0.05
                validity_issues.append("anomalies_not_list")

            valid_anomalies = {a.value for a in AnomalyType}
            invalid_count = sum(1 for a in predicted if a not in valid_anomalies)
            if invalid_count:
                per_item = 0.02
                validity_penalty += min(0.10, invalid_count * per_item)
                validity_issues.append(f"invalid_anomaly_labels={invalid_count}")

            reward, feedback, grader_details = self._grader_anomaly.grade(
                predicted, metadata
            )
            details = {**base_details, **grader_details}
            state["agent_anomalies"] = [
                a for a in predicted
                if isinstance(a, str)
            ]

            anomaly_narratives = []
            for a in state["agent_anomalies"]:
                if a in ANOMALY_LEGAL_BASIS:
                    weight = ANOMALY_SEVERITY.get(AnomalyType(a), 1.0)
                    anomaly_narratives.append(
                        f"{a}: legal basis {ANOMALY_LEGAL_BASIS[a]}, severity_weight={weight:.1f}"
                    )

            details["pre_read"] = (
                f"BOE {manifest.boe_number} at {manifest.port_of_entry}: "
                f"declared USD {int(manifest.declared_value_usd)} vs market "
                f"USD {int(manifest.market_value_usd or 0)}; IEC age "
                f"{manifest.iec_age_months} months."
            )
            details["anomaly_explanation"] = anomaly_narratives
            if anomaly_narratives:
                feedback = f"{feedback} Officer rationale: {'; '.join(anomaly_narratives)}."

        elif step_num == 2:
            # Tasks 2 & 3: assign_channel
            channel_str = action.get("channel", "")
            if channel_str not in {"GREEN", "ORANGE", "RED"}:
                validity_penalty += 0.08
                validity_issues.append("invalid_channel")
            reward, feedback, grader_details = self._grader_channel.grade(
                channel_str,
                metadata,
                agent_anomalies=state["agent_anomalies"],
            )
            details = {**base_details, **grader_details}
            # Fix #6: store for future use
            state["agent_channel"] = channel_str

            if details.get("correct") == "RED" and details.get("assigned") == "ORANGE":
                borderline_penalty = 0.05
                shaping_penalty += borderline_penalty

            details["deliberation"] = (
                f"Anomalies flagged: {state['agent_anomalies']}. "
                f"Assigned {details.get('assigned')} against expected {details.get('correct')}."
            )
            details["escalation"] = {
                "escalate_to_superintendent": details.get("assigned") == "RED",
                "follow_up_within_48h": details.get("assigned") in ("ORANGE", "RED"),
            }

            # Reward coherent progression: good anomaly detection improves policy signal.
            if state["step_rewards"]:
                anomaly_quality = state["step_rewards"][0]
                shaping_bonus += 0.05 * anomaly_quality

        elif step_num == 3:
            # Task 3: draft_scn
            notice_text = action.get("notice_text", "") or action.get("scn_text", "")
            reward, feedback, grader_details = self._grader_scn.grade(
                notice_text,
                manifest,
                metadata,
                agent_anomalies=state["agent_anomalies"],
            )
            details = {**base_details, **grader_details}

            text = notice_text.lower()
            generic_markers = ["under applicable provisions", "as per law", "the aforesaid importer"]
            has_generic = sum(1 for m in generic_markers if m in text) >= 2
            has_specifics = any(str(int(v)) in notice_text for v in [manifest.declared_value_usd, manifest.declared_weight_kg, manifest.iec_age_months])
            if has_generic and not has_specifics:
                shaping_penalty += 0.12
                details["template_penalty"] = 0.12

            contradictions: list[str] = []
            if "severe_undervaluation" in state["agent_anomalies"] and ("section 14" not in text and "valuation" not in text):
                contradictions.append("undervaluation_without_section14_or_valuation_basis")
            if state.get("agent_channel") == "RED" and not any(k in text for k in ["physical inspection", "detention", "seizure"]):
                contradictions.append("red_channel_without_enforcement_linkage")
            if contradictions:
                shaping_penalty += 0.08
                details["scn_contradictions"] = contradictions

            details["scn_pre_plan"] = {
                "likely_sections": ["14", "111", "112", "114A"],
                "must_cite_figures": {
                    "declared_value_usd": int(manifest.declared_value_usd),
                    "market_value_usd": int(manifest.market_value_usd or 0),
                    "declared_weight_kg": int(manifest.declared_weight_kg),
                    "iec_age_months": manifest.iec_age_months,
                },
            }

            # Trajectory-level shaping: SCN should build on earlier decisions.
            if len(state["step_rewards"]) >= 2:
                anomaly_quality = state["step_rewards"][0]
                channel_quality = state["step_rewards"][1]
                shaping_bonus += 0.05 * ((anomaly_quality + channel_quality) / 2)
                if anomaly_quality >= 0.8 and channel_quality >= 0.8:
                    shaping_bonus += 0.05

        else:
            details = dict(base_details)
            feedback = "Unexpected step number."
            reward = 0.0

        base_reward = reward
        reward = max(0.0, min(1.0, reward + shaping_bonus - shaping_penalty - validity_penalty))
        state["step_rewards"].append(round(reward, 4))
        details["expected_task"] = expected_task
        details["declared_task"] = declared_task
        details["shaping_bonus"] = round(shaping_bonus, 4)
        details["shaping_penalty"] = round(shaping_penalty, 4)
        details["validity_penalty"] = round(validity_penalty, 4)
        details["validity_issues"] = validity_issues
        details["rubric"] = {
            "base_reward": round(base_reward, 4),
            "task_alignment": round(max(0.0, 1.0 - (0.10 if declared_task != expected_task else 0.0)), 4),
            "action_validity": round(max(0.0, 1.0 - validity_penalty), 4),
            "trajectory_consistency": round(max(0.0, 1.0 - shaping_penalty), 4),
            "final_reward": round(reward, 4),
        }
        details["score_anatomy"] = {
            "base_reward": round(base_reward, 4),
            "trajectory_bonus": round(shaping_bonus, 4),
            "consistency_or_alignment_penalty": round(shaping_penalty - borderline_penalty, 4),
            "borderline_penalty": round(borderline_penalty, 4),
            "validity_penalty": round(validity_penalty, 4),
            "final_reward": round(reward, 4),
        }

        decision_entry = {
            "step": step_num,
            "task": task,
            "expected_task": expected_task,
            "reward": round(reward, 4),
            "action": action,
            "rubric": details.get("rubric", {}),
        }
        state["decision_log"].append(decision_entry)
        state["last_step_explain"] = {
            "feedback": feedback,
            "details": details,
            "decision_entry": decision_entry,
        }

        state["cumulative_reward"] += reward
        done = step_num >= state["max_steps"]
        state["done"] = done
        if done:
            self._is_active = False

        return StepResponse(
            reward=round(reward, 4),
            feedback=feedback,
            details=details,
            done=done,
            step=step_num,
            cumulative_reward=round(state["cumulative_reward"], 4),
        )

    def get_state(self) -> EnvironmentState:
        """Return current environment state snapshot."""
        if not self._episode_state:
            return EnvironmentState(
                episode_id=None,
                task_name=None,
                step=0,
                max_steps=0,
                done=True,
                cumulative_reward=0.0,
                manifest=None,
            )
        s = self._episode_state
        return EnvironmentState(
            episode_id=s.get("episode_id"),
            trace_id=s.get("trace_id"),
            task_name=s.get("task_name"),
            step=s.get("current_step", 0),
            max_steps=s.get("max_steps", 0),
            done=s.get("done", True),
            cumulative_reward=round(s.get("cumulative_reward", 0.0), 4),
            manifest=s.get("manifest"),
            decision_log=s.get("decision_log", []),
        )

    @property
    def state(self) -> CustomsState:
        """OpenEnv-compatible state property."""
        s = self.get_state()
        return CustomsState(
            episode_id=s.episode_id,
            step_count=s.step,
            task_name=s.task_name,
            max_steps=s.max_steps,
            done=s.done,
            cumulative_reward=s.cumulative_reward,
        )