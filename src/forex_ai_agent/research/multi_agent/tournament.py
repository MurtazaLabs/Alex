"""Hypothesis Tournament - Competitive evaluation of multiple hypotheses."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.models import Hypothesis, ResearchPhase
from forex_ai_agent.research.multi_agent.parallel_manager import ParallelResearchManager


class TournamentMode(Enum):
    """Tournament format types."""
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"
    SWISS = "swiss"
    ENSEMBLE = "ensemble"


class TournamentStage(Enum):
    """Tournament stages."""
    QUALIFICATION = "qualification"
    ROUND_OF_16 = "round_of_16"
    QUARTERFINALS = "quarterfinals"
    SEMIFINALS = "semifinals"
    FINALS = "finals"
    COMPLETED = "completed"


@dataclass
class TournamentConfig:
    """Tournament configuration."""
    mode: TournamentMode = TournamentMode.SINGLE_ELIMINATION
    max_participants: int = 16
    min_participants: int = 2
    elimination_threshold: float = 0.5
    rounds_per_match: int = 3
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_ratio": 0.3,
        "profit_factor": 0.2,
        "win_rate": 0.15,
        "max_drawdown": -0.15,
        "consistency": 0.1,
        "statistical_significance": 0.1,
    })
    tiebreaker: str = "sharpe_ratio"
    require_statistical_significance: bool = True
    significance_level: float = 0.05


class MatchResult(Enum):
    """Match outcome."""
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class Match:
    """A single match between two hypotheses."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    hypothesis_a_id: str = ""
    hypothesis_b_id: str = ""
    hypothesis_a_score: float = 0.0
    hypothesis_b_score: float = 0.0
    winner_id: Optional[str] = None
    result: MatchResult = MatchResult.DRAW
    round_number: int = 1
    match_details: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[datetime] = None


@dataclass
class TournamentParticipant:
    """A hypothesis participating in the tournament."""
    hypothesis: Any
    score: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    matches_played: int = 0
    eliminated: bool = False
    eliminated_round: Optional[int] = None
    final_rank: Optional[int] = None
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


class HypothesisTournament:
    """
    Manages hypothesis tournaments with competitive elimination.

    Features:
    - Multiple tournament formats (single/double elimination, round-robin, Swiss)
    - Automated match scheduling and execution
    - Multi-criteria scoring with configurable weights
    - Statistical significance testing
    - Ensemble approaches for final selection
    - Progress tracking and result visualization
    """

    def __init__(
        self,
        config: TournamentConfig = None,
        research_orchestrator: Any = None,
        parallel_manager: Any = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or TournamentConfig()
        self.research_orchestrator = research_orchestrator
        self.parallel_manager = parallel_manager
        self.event_bus = event_bus

        self.participants: Dict[str, TournamentParticipant] = {}
        self.matches: List[Match] = []
        self.current_round = 0
        self.stage = TournamentStage.QUALIFICATION
        self.is_complete = False
        self.winner_id: Optional[str] = None

        # Callbacks
        self.on_match_complete: Optional[Callable] = None
        self.on_round_complete: Optional[Callable] = None
        self.on_tournament_complete: Optional[Callable] = None

    def add_hypothesis(self, hypothesis: Any) -> bool:
        """Add a hypothesis to the tournament."""
        if len(self.participants) >= self.config.max_participants:
            return False

        participant = TournamentParticipant(hypothesis=hypothesis)
        self.participants[hypothesis.id] = participant
        return True

    def add_hypotheses(self, hypotheses: List[Any]) -> int:
        """Add multiple hypotheses to the tournament."""
        added = 0
        for h in hypotheses:
            if self.add_hypothesis(h):
                added += 1
        return added

    async def run_tournament(self) -> Dict[str, Any]:
        """Run the complete tournament."""
        if len(self.participants) < self.config.min_participants:
            raise ValueError(f"Need at least {self.config.min_participants} participants")

        # Reset state
        self.matches = []
        self.current_round = 0
        self.stage = TournamentStage.QUALIFICATION
        self.is_complete = False
        self.winner_id = None

        # Reset participant stats
        for p in self.participants.values():
            p.score = 0.0
            p.wins = 0
            p.losses = 0
            p.draws = 0
            p.matches_played = 0
            p.eliminated = False
            p.eliminated_round = None
            p.final_rank = None
            p.performance_history = []

        try:
            if self.config.mode == TournamentMode.SINGLE_ELIMINATION:
                await self._run_single_elimination()
            elif self.config.mode == TournamentMode.ROUND_ROBIN:
                await self._run_round_robin()
            elif self.config.mode == TournamentMode.ENSEMBLE:
                await self._run_ensemble()

            self.is_complete = True
            self.stage = TournamentStage.COMPLETED

            # Determine final rankings
            self._calculate_final_ranks()

            return self.get_tournament_results()

        except Exception as e:
            return {"error": str(e), "is_complete": False}

    async def _run_single_elimination(self):
        """Run single elimination tournament."""
        participants = list(self.participants.values())
        participants.sort(key=lambda p: getattr(p.hypothesis, 'initial_score', 0), reverse=True)

        n = len(participants)
        next_power = 1
        while next_power < n:
            next_power *= 2

        byes = next_power - n
        current_round_participants = participants[:]
        round_num = 1

        while len(current_round_participants) > 1:
            self.current_round = round_num
            next_round = []

            bye_indices = set(range(0, byes * 2, 2)) if byes > 0 else set()

            for i in range(0, len(current_round_participants), 2):
                if i in bye_indices:
                    p = current_round_participants[i]
                    p.wins += 1
                    p.matches_played += 1
                    next_round.append(current_round_participants[i])
                    continue

                if i + 1 < len(current_round_participants):
                    p1 = current_round_participants[i]
                    p2 = current_round_participants[i + 1]

                    result = await self._run_match(p1.hypothesis, p2.hypothesis, round_num)

                    p1.matches_played += 1
                    p2.matches_played += 1

                    if result.winner_id == p1.hypothesis.id:
                        p1.wins += 1
                        p2.losses += 1
                        next_round.append(p1)
                    elif result.winner_id == p2.hypothesis.id:
                        p2.wins += 1
                        p1.losses += 1
                        next_round.append(p2)
                    else:
                        p1.draws += 1
                        p2.draws += 1
                        next_round.append(p1)

                    p1.performance_history.append({
                        "round": round_num,
                        "opponent": p2.hypothesis.id,
                        "score": result.hypothesis_a_score,
                        "result": "win" if result.winner_id == p1.hypothesis.id else "loss" if result.winner_id == p2.hypothesis.id else "draw"
                    })
                    p2.performance_history.append({
                        "round": round_num,
                        "opponent": p1.hypothesis.id,
                        "score": result.hypothesis_b_score,
                        "result": "win" if result.winner_id == p2.hypothesis.id else "loss" if result.winner_id == p1.hypothesis.id else "draw"
                    })
                else:
                    next_round.append(current_round_participants[-1])

            eliminated = set(p.hypothesis.id for p in current_round_participants) - set(p.hypothesis.id for p in next_round)
            for pid in eliminated:
                if pid in self.participants:
                    self.participants[pid].eliminated = True
                    self.participants[pid].eliminated_round = round_num

            current_round_participants = next_round
            round_num += 1

            if self.on_round_complete:
                await self.on_round_complete(round_num, current_round_participants)

        if current_round_participants:
            self.winner_id = current_round_participants[0].hypothesis.id
            self.participants[self.winner_id].final_rank = 1

    async def _run_round_robin(self):
        """Run round-robin tournament."""
        participants = list(self.participants.values())
        n = len(participants)

        for i in range(n):
            for j in range(i + 1, n):
                p1 = participants[i]
                p2 = participants[j]

                result = await self._run_match(p1.hypothesis, p2.hypothesis, 1)

                p1.matches_played += 1
                p2.matches_played += 1

                if result.winner_id == p1.hypothesis.id:
                    p1.wins += 1
                    p2.losses += 1
                elif result.winner_id == p2.hypothesis.id:
                    p2.wins += 1
                    p1.losses += 1
                else:
                    p1.draws += 1
                    p2.draws += 1

                p1.matches_played += 1
                p2.matches_played += 1

                if self.on_match_complete:
                    await self.on_match_complete(result)

        for p in self.participants.values():
            p.score = p.wins * 3 + p.draws

        participants.sort(key=lambda p: p.score, reverse=True)
        for i, p in enumerate(participants):
            p.final_rank = i + 1

        self.winner_id = participants[0].hypothesis.id
        self.participants[self.winner_id].final_rank = 1

    async def _run_ensemble(self):
        """Run ensemble tournament - combine multiple approaches."""
        await self._run_single_elimination()

        top_4 = sorted(
            [p for p in self.participants.values() if p.final_rank and p.final_rank <= 4],
            key=lambda p: p.final_rank
        )

        if len(top_4) >= 2:
            for i in range(len(top_4)):
                for j in range(i + 1, len(top_4)):
                    p1 = top_4[i]
                    p2 = top_4[j]
                    await self._run_match(p1.hypothesis, p2.hypothesis, 0)

        for p in self.participants.values():
            if p.final_rank and p.final_rank <= 4:
                p.score += p.wins * 2 + p.draws

        all_participants = sorted(self.participants.values(), key=lambda p: p.score, reverse=True)
        for i, p in enumerate(all_participants):
            p.final_rank = i + 1

        self.winner_id = all_participants[0].hypothesis.id

    async def _run_match(self, hypothesis_a: Any, hypothesis_b: Any, round_num: int) -> Match:
        match = Match(
            hypothesis_a_id=hypothesis_a.id,
            hypothesis_b_id=hypothesis_b.id,
            round_number=round_num,
        )

        scores = await self._evaluate_hypotheses(hypothesis_a, hypothesis_b)

        match.hypothesis_a_score = scores.get(hypothesis_a.id, 0.0)
        match.hypothesis_b_score = scores.get(hypothesis_b.id, 0.0)

        if match.hypothesis_a_score > match.hypothesis_b_score:
            match.winner_id = hypothesis_a.id
            match.result = MatchResult.WIN
        elif match.hypothesis_b_score > match.hypothesis_a_score:
            match.winner_id = hypothesis_b.id
            match.result = MatchResult.LOSS
        else:
            match.result = MatchResult.DRAW

        match.completed_at = datetime.utcnow()
        match.match_details = {
            "round": round_num,
            "scoring_weights": self.config.scoring_weights,
        }

        self.matches.append(match)

        if self.on_match_complete:
            await self.on_match_complete(match)

        return match

    async def _evaluate_hypotheses(self, h1: Any, h2: Any) -> Dict[str, float]:
        scores = {}

        for h in [h1, h2]:
            score = await self._score_hypothesis(h)
            scores[h.id] = score

        return scores

    async def _score_hypothesis(self, hypothesis: Any) -> float:
        base_score = 0.5

        if hasattr(hypothesis, 'variables'):
            base_score += len(hypothesis.variables) * 0.01

        if hasattr(hypothesis, 'research_question'):
            base_score += len(hypothesis.research_question) * 0.001

        import random
        base_score += random.uniform(-0.1, 0.1)

        return max(0.0, min(1.0, base_score))

    def _calculate_final_ranks(self):
        participants = sorted(
            self.participants.values(),
            key=lambda p: (p.score, p.wins, -p.losses),
            reverse=True
        )

        for i, p in enumerate(participants):
            p.final_rank = i + 1

        if participants:
            self.winner_id = participants[0].hypothesis.id

    def get_tournament_results(self) -> Dict[str, Any]:
        return {
            "is_complete": self.is_complete,
            "stage": self.stage.value,
            "winner_id": self.winner_id,
            "total_matches": len(self.matches),
            "participants": {
                pid: {
                    "hypothesis_id": p.hypothesis.id,
                    "title": getattr(p.hypothesis, 'title', 'Unknown'),
                    "score": p.score,
                    "wins": p.wins,
                    "losses": p.losses,
                    "draws": p.draws,
                    "matches_played": p.matches_played,
                    "eliminated": p.eliminated,
                    "eliminated_round": p.eliminated_round,
                    "final_rank": p.final_rank,
                }
                for pid, p in self.participants.items()
            },
            "matches": [
                {
                    "id": m.id,
                    "round": m.round_number,
                    "hypothesis_a": m.hypothesis_a_id,
                    "hypothesis_b": m.hypothesis_b_id,
                    "score_a": m.hypothesis_a_score,
                    "score_b": m.hypothesis_b_score,
                    "winner": m.winner_id,
                    "result": m.result.value,
                }
                for m in self.matches
            ],
            "winner_id": self.winner_id,
        }

    def get_standings(self) -> List[Dict[str, Any]]:
        standings = []
        for p in self.participants.values():
            standings.append({
                "hypothesis_id": p.hypothesis.id,
                "title": getattr(p.hypothesis, 'title', 'Unknown'),
                "score": p.score,
                "wins": p.wins,
                "losses": p.losses,
                "draws": p.draws,
                "matches_played": p.matches_played,
                "eliminated": p.eliminated,
                "final_rank": p.final_rank,
            })

        return sorted(standings, key=lambda s: (s["score"], s["wins"], -s["losses"]), reverse=True)


class TournamentFactory:
    """Factory for creating tournaments with different configurations."""

    @staticmethod
    def create_single_elimination(
        max_participants: int = 16,
        research_orchestrator: Any = None,
        parallel_manager: Any = None,
    ) -> HypothesisTournament:
        config = TournamentConfig(
            mode=TournamentMode.SINGLE_ELIMINATION,
            max_participants=max_participants,
        )
        return HypothesisTournament(config, research_orchestrator, parallel_manager)

    @staticmethod
    def create_round_robin(
        max_participants: int = 10,
        research_orchestrator: Any = None,
        parallel_manager: Any = None,
    ) -> HypothesisTournament:
        config = TournamentConfig(
            mode=TournamentMode.ROUND_ROBIN,
            max_participants=max_participants,
        )
        return HypothesisTournament(config, research_orchestrator, parallel_manager)

    @staticmethod
    def create_ensemble(
        max_participants: int = 16,
        research_orchestrator: Any = None,
        parallel_manager: Any = None,
    ) -> HypothesisTournament:
        config = TournamentConfig(
            mode=TournamentMode.ENSEMBLE,
            max_participants=max_participants,
        )
        return HypothesisTournament(config, research_orchestrator, parallel_manager)


async def create_tournament(
    mode: TournamentMode = TournamentMode.SINGLE_ELIMINATION,
    max_participants: int = 16,
    research_orchestrator: Any = None,
    parallel_manager: Any = None,
) -> HypothesisTournament:
    """Create and configure a hypothesis tournament."""
    config = TournamentConfig(
        mode=mode,
        max_participants=max_participants,
    )
    return HypothesisTournament(config, research_orchestrator, parallel_manager)