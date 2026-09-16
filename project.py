
import argparse
import os
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    "run": ("__main__", []),
    "demo": ("semantic_workflow", ["--demo"]),
    "history-demo": ("workbench", ["demo"]),
    "trace": ("trace_report", []),
    "view": ("runtime_view", []),
    "evaluate-status": ("evaluation_status", []),
    "business": ("business_batch", []),
    "score": ("business_score", []),
    "retry-check": ("retry_compare", []),
    "semantic-check": ("semantic_compare", []),
    "sdk-check": ("sdk_controller", []),
    "sdk-fault-check": ("sdk_fault_compare", []),
    "sdk-resume-check": ("sdk_resume_check", []),
}

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Agent project entry point"
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    module, defaults = COMMANDS[args.command]
    module_path = root / "src" / "knowledge_agents" / "harness" / (module + ".py")
    if not module_path.is_file():
        parser.error("Missing module: " + str(module_path))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable, "-m",
            "knowledge_agents.harness." + module,
            *defaults, *args.arguments,
        ],
        cwd=root,
        env=env,
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
