from __future__ import annotations

import argparse
import json

from adaptive_agent.runtime import AgentRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive Repository Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the Q-learning routing policy")
    train.add_argument("--episodes", type=int, default=160)

    run = subparsers.add_parser("run", help="Run one benchmark task")
    run.add_argument("task", choices=("division-by-zero", "decimal-average"))
    run.add_argument("--epsilon", type=float, default=0.0)

    custom = subparsers.add_parser("run-custom", help="Run an Issue against a local repository")
    custom.add_argument("repository", help="Path to the local repository")
    custom.add_argument("issue", help="Issue text or development task")
    custom.add_argument("--search-term", default="")
    custom.add_argument("--test-target", default="tests")
    custom.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip the approval interrupt; use only with a disposable repository",
    )

    subparsers.add_parser("policy", help="Print learned Q-values")
    return parser


def main() -> None:
    args = _parser().parse_args()
    runtime = AgentRuntime()

    if args.command == "train":
        history = runtime.train(args.episodes)
        completed = sum(item["status"] == "completed" for item in history)
        print(json.dumps(history[-10:], indent=2))
        print(f"Completed {completed}/{len(history)} episodes")
        return

    if args.command == "run":
        outcome = runtime.run_task(args.task, auto_approve=True, epsilon=args.epsilon)
        print(json.dumps(outcome["result"], indent=2, default=str))
        return

    if args.command == "run-custom":
        outcome = runtime.run_custom_task(
            args.repository,
            args.issue,
            search_term=args.search_term,
            test_target=args.test_target,
            auto_approve=args.auto_approve,
        )
        print(json.dumps(outcome["result"], indent=2, default=str))
        return

    print(json.dumps(runtime.policy_rows(), indent=2))


if __name__ == "__main__":
    main()
