from adaptive_agent.rl import QLearningPolicy


def test_policy_prefers_best_learned_action() -> None:
    policy = QLearningPolicy(seed=1)
    state = {"context_ready": True, "plan_ready": False, "approved": False, "patch_applied": False}
    observation = policy.observation(state)
    policy.update(observation, "plan", 1.0, observation, done=True)

    assert policy.choose(state, epsilon=0.0) == "plan"


def test_policy_bootstrap_sequence_is_safe() -> None:
    policy = QLearningPolicy()
    state = {"context_ready": False, "plan_ready": False, "approved": False, "patch_applied": False}

    assert policy.choose(state, epsilon=0.0) == "retrieve"
