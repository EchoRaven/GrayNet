"""Execute bounded host commands without shell interpretation."""

import subprocess


class CommandError(RuntimeError):
    pass


def run_process(
    args: list[str], *, timeout: int = 30, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommandError(f"command failed: {' '.join(args)}: {exc}") from exc
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1200:]
        raise CommandError(f"command failed ({result.returncode}): {' '.join(args)}: {detail}")
    return result
