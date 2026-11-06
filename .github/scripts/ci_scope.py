"""Select the ordinary CP311 or explicitly requested full CI scope."""

import json
import os
from pathlib import Path
import subprocess


PYTHON_TAGS = ["cp310", "cp311", "cp312", "cp313", "cp314"]


def is_documentation(path):
    if path.startswith(("doc/", "assets/readme/")):
        return True
    return path.endswith((".md", ".rst")) and not path.startswith(
        ("hotpot/", "tests/", "tools/", "molecules/")
    )


def changed_range(event_name, event):
    if event_name == "pull_request":
        head = event["pull_request"]["head"]["sha"]
        base = subprocess.check_output(
            ["git", "merge-base", event["pull_request"]["base"]["sha"], head],
            text=True,
        ).strip()
        return base, head
    if event_name == "push":
        base, head = event["before"], event["after"]
        # A first push or a force-push may have no available old commit.
        if set(base) != {"0"} and subprocess.run(
            ["git", "cat-file", "-e", base + "^{commit}"],
            stderr=subprocess.DEVNULL,
        ).returncode == 0:
            return base, head
    return None


def main():
    event_name = os.environ["GITHUB_EVENT_NAME"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    full = os.environ.get("HOTPOT_CI_FULL_MATRIX") == "true"
    revisions = changed_range(event_name, event)
    files = None
    if revisions is not None:
        files = subprocess.check_output([
            "git", "diff", "--no-renames", "--name-only", "-z", *revisions,
        ]).decode().split("\0")[:-1]
    core = (
        full
        or event_name not in ("push", "pull_request")
        or files is None
        or not all(is_documentation(path) for path in files)
    )
    if not core:
        subprocess.run(["git", "diff", "--check", *revisions], check=True)
    tags = PYTHON_TAGS if full else ["cp311"]
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"core={str(core).lower()}\n")
        output.write("python_tags=" + json.dumps(tags, separators=(",", ":")) + "\n")
    print(json.dumps({"core": core, "python_tags": tags, "changed_files": files}))


if __name__ == "__main__":
    main()
