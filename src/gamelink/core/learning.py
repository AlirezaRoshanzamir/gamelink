from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, TypeVar, cast, override

from gamelink.core.game import (
    Action,
    DecisionProducer,
    DecisionSelector,
    Game,
    Player,
    Probabilistic,
    Readonly,
    State,
)
from gamelink.utils import Timeline


class OptimalPlayer[
    TGame: Game[Any, Any, Any],
    TState: State,
    TAction: Action[Any],
    TPlayer: Player[Any, Any],
](Player[TState, TAction]):
    def __init__(
        self,
        possible_actions_generator: Callable[[TPlayer, TState], Iterable[TAction]],
        q_function: QFunction[TPlayer, TState, TAction],
        decision_selector: DecisionSelector | None = None,
        **kwargs: Any,  # noqa: ANN401  # For cooperative multiple inheritance
    ) -> None:
        super().__init__(decision_selector=decision_selector, **kwargs)
        self._possible_actions_generator = possible_actions_generator
        self._q_function = q_function

    @override
    def act(self, state: TState) -> TAction:
        each_action_value: dict[TAction, float] = defaultdict(lambda: 0)

        for action in self._possible_actions_generator(cast("TPlayer", self), state):
            each_action_value[action] = self._q_function.q_evaluate(
                cast("TPlayer", self), state, action
            )

        best_action = max(
            each_action_value.keys(),
            key=lambda key: each_action_value[key],
        )

        return self.select_decision(
            decisions=Probabilistic.many_from_mapping(
                {
                    action: 1.0 if action is best_action else 0.0
                    for action, score in each_action_value.items()
                }
            ),
        )


class FOVFunction[
    TGame: Game[Any, Any, Any],
    TPlayer: Player[Any, Any],
    TState: State,
](ABC):
    @abstractmethod
    def fov_evaluate(self, game: TGame, player: TPlayer, state: TState) -> float:
        pass


class VFunction[
    TPlayer: Player[Any, Any],
    TState: State,
](FOVFunction[Game[Any, Any, Any], TPlayer, TState]):
    @abstractmethod
    def v_evaluate(self, player: TPlayer, state: TState) -> float:
        pass

    @override
    def fov_evaluate(
        self, game: Game[Any, Any, Any], player: TPlayer, state: TState
    ) -> float:
        return self.v_evaluate(player, state)


class FOQFunction[
    TGame: Game[Any, Any, Any],
    TPlayer: Player[Any, Any],
    TState: State,
    TAction: Action[Any],
](ABC):
    @abstractmethod
    def foq_evaluate(
        self, game: TGame, player: TPlayer, state: TState, action: TAction
    ) -> float:
        pass


class QFunction[TPlayer: Player[Any, Any], TState: State, TAction: Action[Any]](
    FOQFunction[Game[Any, Any, Any], TPlayer, TState, TAction]
):
    @abstractmethod
    def q_evaluate(self, player: TPlayer, state: TState, action: TAction) -> float:
        pass

    @override
    def foq_evaluate(
        self, game: Game[Any, Any, Any], player: TPlayer, state: TState, action: TAction
    ) -> float:
        return self.q_evaluate(player, state, action)


class FOQFunctionAsVFunction[
    TGame: Game[Any, Any, Any],
    TPlayer: Player[Any, Any],
    TState: State,
    TAction: Action[Any],
](VFunction[TPlayer, TState]):
    def __init__(
        self,
        foq_function: FOQFunction[TGame, TPlayer, TState, TAction],
        game: Readonly[TGame],
    ) -> None:
        self._game = game
        self._foq_function = foq_function

    @override
    def v_evaluate(self, player: TPlayer, state: TState) -> float:
        each_action_value: dict[Action[Any], float] = defaultdict(lambda: 0)
        for action in self._game.possible_actions_for_player(player, state):
            each_action_value[action] = self._foq_function.foq_evaluate(
                self._game, player, state, action
            )

        best_action = max(
            each_action_value.keys(),
            key=lambda key: each_action_value[key],
        )

        return each_action_value[best_action]


class FOVFunctionAsFOQFunction[
    TGame: Game[Any, Any, Any],
    TPlayer: Player[Any, Any],
    TState: State,
](FOQFunction[TGame, TPlayer, TState, Action[Any]]):
    def __init__(
        self,
        fov_function: FOVFunction[TGame, TPlayer, TState],
    ) -> None:
        self._fov_function = fov_function

    @override
    def foq_evaluate(
        self, game: TGame, player: TPlayer, state: TState, action: Action[Any]
    ) -> float:
        # TODO: Use the Game class interfaces for simulating the game.
        action.do(game)
        value = self._fov_function.fov_evaluate(game, player, state)
        action.revert(game)
        return value


class FOQFunctionAsQFunction[
    TGame: Game[Any, Any, Any],
    TPlayer: Player[Any, Any],
    TState: State,
    TAction: Action[Any],
](QFunction[TPlayer, TState, TAction]):
    def __init__(
        self,
        foq_function: FOQFunction[TGame, TPlayer, TState, TAction],
        possible_games_generator: Callable[[TState], Iterable[Probabilistic[TGame]]],
    ) -> None:
        self._foq_function = foq_function
        self._possible_games_generator = possible_games_generator

    @override
    def q_evaluate(self, player: TPlayer, state: TState, action: TAction) -> float:
        total_value = 0.0
        for probabilistic_game in self._possible_games_generator(state):
            game, probability = probabilistic_game.event, probabilistic_game.probability
            total_value = probability * self._foq_function.foq_evaluate(
                game, player, state, action
            )
        return total_value


class PlayerSpy[TState: State, TAction: Action[Any]](Player[TState, TAction]):
    def __init__(
        self,
        underlying_player: Player[TState, TAction],
        decision_selector: DecisionSelector | None = None,
    ) -> None:
        super().__init__(decision_selector)
        self._first_state_action_pair: tuple[TState, TAction] | None = None
        self._underlying_player = underlying_player

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying_player, name)

    @property
    def first_state_action_pair(self) -> tuple[TState, TAction] | None:
        return self._first_state_action_pair

    def reset(self) -> None:
        self._first_state_action_pair = None

    def act(self, state: TState) -> TAction:
        action = self._underlying_player.act(state)
        if self._first_state_action_pair is None:
            self._first_state_action_pair = (state, action)
        return action


class BruteForceFOVFunction[
    TGame: Game[Any, Any, Any],
    TPlayer: Player[Any, Any],
    TState: State,
](
    FOVFunction[TGame, TPlayer, TState],
):
    def __init__(
        self,
        players_replacer: Callable[[TPlayer, TPlayer], TPlayer],
        node_factory: BacktrackingDecisionSelectorNodeFactory[
            BacktrackingDecisionSelectorNode
        ],
        max_depth: int | None = None,
    ) -> None:
        super().__init__()
        self._node_factory = node_factory
        self._players_replacer = players_replacer
        self._terminal_v_function: Callable[[TGame, TPlayer], float] = (
            lambda game, player: game.final_score_for_player(player)
        )
        self._max_depth = max_depth or sys.maxsize

    @override
    def fov_evaluate(self, game: TGame, player: TPlayer, state: TState) -> float:
        if game.finished:
            return self._terminal_v_function(game, player)

        player_replacements = {
            replacing_player: self._players_replacer(player, replacing_player)
            for replacing_player in game.players
            if replacing_player is not player
        }
        current_player_replacement = self._players_replacer(player, player)
        player_replacements[player] = current_player_replacement

        decision_selector = BacktrackingDecisionSelector(
            node_factory=self._node_factory,
            evaluating_producer_predicate=lambda producer: (
                producer is current_player_replacement
            ),
        )
        current_depth = 0

        with game.simulate(
            player_replacements=player_replacements,
            decision_selector=decision_selector,
        ):
            while True:
                while not game.finished and current_depth < self._max_depth:
                    decision_selector.checkpoint_current()
                    game.step_forward()
                    current_depth += 1

                value = self._terminal_v_function(game, current_player_replacement)
                decision_selector.add_value(value)

                while decision_selector.any_checkpoint():
                    game.step_backward()
                    decision_selector.pop_then_seek_checkpoint()
                    current_depth -= 1

                    if decision_selector.any_remaining_decision_in_future():
                        break

                if not decision_selector.any_remaining_decision_in_future():
                    break

        return decision_selector.current_value


class BacktrackingDecisionSelector[TNode: BacktrackingDecisionSelectorNode](
    DecisionSelector
):
    def __init__(
        self,
        node_factory: BacktrackingDecisionSelectorNodeFactory[TNode],
        evaluating_producer_predicate: Callable[[DecisionProducer], bool],
    ) -> None:
        super().__init__()
        self._node_factory = node_factory
        self._evaluating_producer_predicate = evaluating_producer_predicate
        self._root_visitor = node_factory.create(
            self._evaluating_producer_predicate, None, None
        )
        self._path_timeline: Timeline[TNode] = Timeline(
            root=self._root_visitor,
        )

    @property
    def current_value(self) -> float:
        return self._path_timeline.current.aggregated_value

    def value_for_first_node_of(self, producer: DecisionProducer) -> float:
        return next(
            node.aggregated_value
            for node in self._path_timeline.future
            if node.last_producer is producer
        )

    def any_checkpoint(self) -> bool:
        return self._path_timeline.any_checkpoint()

    def checkpoint_current(self) -> None:
        self._path_timeline.checkpoint_current()

    def add_value(self, value: float) -> None:
        self._path_timeline.current.add_value(value)

    def pop_then_seek_checkpoint(self) -> None:
        checkpoint_index = self._path_timeline.pop_checkpoint()

        value = self._path_timeline.current.aggregated_value
        next_node = None
        for i, this_node in enumerate(
            self._path_timeline.reversed(inclusive_to=checkpoint_index)
        ):
            if next_node is None:
                next_node = this_node
                continue
            if next_node.has_next():
                break
            this_node.add_value(value)
            value = this_node.aggregated_value
            next_node = this_node

        self._path_timeline.seek(checkpoint_index)

    def any_remaining_decision_in_future(self) -> bool:
        return any(selector.has_next() for selector in self._path_timeline.future)

    @override
    def select_index_hook[TDecision](
        self,
        producer: DecisionProducer,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        if not self._path_timeline.future:
            self._path_timeline.append(
                self._node_factory.create(
                    self._evaluating_producer_predicate,
                    producer,
                    self._path_timeline.current,
                )
            )

        self._path_timeline.step_forward()

        if (
            not self.any_remaining_decision_in_future()
            and self._path_timeline.current.has_next()
        ):
            self._path_timeline.clear_future()
            self._path_timeline.current.to_next()

        return self._path_timeline.current.select_index(producer, decisions, title)


class BacktrackingDecisionSelectorNode(DecisionSelector):
    def __init__(self, should_continue: Callable[[], bool] | None = None) -> None:
        super().__init__()
        self._current_selecting_decision_index: int = 0
        self._should_continue = should_continue or (lambda: True)

    @property
    def aggregated_value(self) -> float:
        return 0.0

    def add_value(self, value: float) -> None:
        pass

    def to_next(self) -> None:
        if not self.has_next():
            msg = "No remaining decision to move to."
            raise RuntimeError(msg)

        self._current_selecting_decision_index += 1

    def has_next(self) -> bool:
        return self._should_continue() and (
            self._current_selecting_decision_index < self.last_number_of_decisions - 1
        )

    @override
    def select_index_hook[TDecision](
        self,
        producer: DecisionProducer,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        return self._current_selecting_decision_index


class GenericBacktrackingDecisionSelectorNode(BacktrackingDecisionSelectorNode):
    def __init__(
        self,
        *,
        initial_value: float | None = None,
        value_aggregator: Callable[[float, float, float | None], float] | None = None,
        initial_alpha: float | None = None,
        alpha_aggregator: Callable[[float, float, float | None], float] | None = None,
        alpha_enabled: bool = False,
        initial_beta: float | None = None,
        beta_aggregator: Callable[[float, float, float | None], float] | None = None,
        beta_enabled: bool = False,
    ) -> None:
        super().__init__(should_continue=self._within_cutoffs)
        self._value = initial_value or 0.0
        self._value_aggregator = value_aggregator or (
            lambda _pre_value, value, _probability: value
        )
        self._alpha = initial_alpha if initial_alpha is not None else -math.inf
        self._alpha_aggregator = alpha_aggregator or (
            lambda alpha, _value, _probability: alpha
        )
        self._alpha_enabled = alpha_enabled
        self._beta = initial_beta if initial_beta is not None else math.inf
        self._beta_aggregator = beta_aggregator or (
            lambda beta, _value, _probability: beta
        )
        self._beta_enabled = beta_enabled
        self._last_decision_probability: float | None = None

    @property
    @override
    def aggregated_value(self) -> float:
        return self._value

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def alpha(self) -> float:
        return self._alpha

    @classmethod
    def create_identity(cls) -> GenericBacktrackingDecisionSelectorNode:
        return GenericBacktrackingDecisionSelectorNode(
            value_aggregator=lambda _pre_value, value, _probability: value,
            initial_beta=math.inf,
            initial_alpha=-math.inf,
            initial_value=0.0,
        )

    @override
    def add_value(self, value: float) -> None:
        self._value = self._value_aggregator(
            self._value, value, self._last_decision_probability
        )

        if self._within_cutoffs():
            self._beta = self._beta_aggregator(
                self._beta, self._value, self._last_decision_probability
            )
            self._alpha = self._alpha_aggregator(
                self._alpha, self._value, self._last_decision_probability
            )

    @override
    @override
    def select_index_hook[TDecision](
        self,
        producer: DecisionProducer,
        decisions: Sequence[Probabilistic[TDecision]],
        title: str | None = None,
    ) -> int:
        selected_index = super().select_index_hook(producer, decisions, title)
        self._last_decision_probability = decisions[selected_index].probability
        return selected_index

    def _within_cutoffs(self) -> bool:
        result = True

        if self._beta_enabled:
            result = result and (self._value < self._beta)

        if self._alpha_enabled:
            result = result and (self._alpha < self._value)

        return result


# Mypy cannot infer the variance of this type parameter, so we use
# TypeVar to specify it explicitly.
TBacktrackingDecisionSelectorNode = TypeVar(
    "TBacktrackingDecisionSelectorNode",
    bound=BacktrackingDecisionSelectorNode,
    covariant=True,
)


class BacktrackingDecisionSelectorNodeFactory(
    ABC, Generic[TBacktrackingDecisionSelectorNode]
):
    @abstractmethod
    def create(
        self,
        evaluating_producer_predicate: Callable[[DecisionProducer], bool],
        current_producer: DecisionProducer | None,
        pre_node: TBacktrackingDecisionSelectorNode | None,
    ) -> TBacktrackingDecisionSelectorNode:
        pass


class MinimaxBacktrackingDecisionSelectorNodeFactory(
    BacktrackingDecisionSelectorNodeFactory[GenericBacktrackingDecisionSelectorNode]
):
    @override
    def create(
        self,
        evaluating_producer_predicate: Callable[[DecisionProducer], bool],
        current_producer: DecisionProducer | None,
        pre_node: GenericBacktrackingDecisionSelectorNode | None,
    ) -> GenericBacktrackingDecisionSelectorNode:
        if current_producer is None or pre_node is None:
            return GenericBacktrackingDecisionSelectorNode.create_identity()

        if evaluating_producer_predicate(current_producer):
            # Max node
            return GenericBacktrackingDecisionSelectorNode(
                initial_value=-math.inf,
                value_aggregator=lambda pre_value, value, probability: max(
                    pre_value, value
                ),
                # initial_beta=pre_node.beta,
                # beta_enabled=True,
                # initial_alpha=pre_node.alpha,
                # alpha_aggregator=max,
                # alpha_enabled=False,
            )
        if not current_producer.stochastic:
            # Min node
            return GenericBacktrackingDecisionSelectorNode(
                initial_value=math.inf,
                value_aggregator=lambda pre_value, value, probability: min(
                    pre_value, value
                ),
                # initial_beta=pre_node.beta,
                # beta_aggregator=min,
                # beta_enabled=False,
                # initial_alpha=pre_node.alpha,
                # alpha_enabled=True,
            )
        # Random event node
        return GenericBacktrackingDecisionSelectorNode(
            initial_value=0,
            value_aggregator=lambda pre_value, value, probability: (
                pre_value + (probability if probability is not None else 1.0) * value
            ),
        )
