#!/usr/bin/env python3
"""Open and hold one regular lock file without following or truncating it."""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import sys
from pathlib import Path
from typing import NoReturn


def fail(message: str, status: int = 1) -> NoReturn:
    print(f"[safe-lock] {message}", file=sys.stderr)
    raise SystemExit(status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--busy-exit", type=int, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if args.fd < 3:
        fail("lock fd must be at least 3", 2)
    if not 1 <= args.busy_exit <= 255:
        fail("--busy-exit must be in [1, 255]", 2)
    if not command:
        fail("a command is required after --", 2)

    lock_path = args.path.absolute()
    if lock_path.name in ("", ".", ".."):
        fail(f"unsafe lock path: {lock_path}", 2)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        parent_fd = os.open(lock_path.parent, parent_flags)
    except OSError as error:
        fail(f"cannot open lock parent {lock_path.parent}: {error}")
    try:
        open_flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
        open_flags |= getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow:
            fail("platform lacks O_NOFOLLOW; refusing an unsafe lock open")
        open_flags |= nofollow
        try:
            opened_fd = os.open(
                lock_path.name,
                open_flags,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as error:
            fail(f"cannot safely open lock {lock_path}: {error}")
        try:
            opened_stat = os.fstat(opened_fd)
            path_stat = os.stat(
                lock_path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(opened_stat.st_mode):
                fail(f"lock is not a regular file: {lock_path}")
            if opened_stat.st_nlink != 1:
                fail(f"lock must have exactly one hard link: {lock_path}")
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                path_stat.st_dev,
                path_stat.st_ino,
            ):
                fail(f"lock identity changed while opening: {lock_path}")
            try:
                fcntl.flock(opened_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fail(f"lock is already held: {lock_path}", args.busy_exit)
            if opened_fd != args.fd:
                os.dup2(opened_fd, args.fd, inheritable=True)
                os.close(opened_fd)
            else:
                os.set_inheritable(opened_fd, True)
        except BaseException:
            try:
                os.close(opened_fd)
            except OSError:
                pass
            raise
    finally:
        os.close(parent_fd)

    os.execvpe(command[0], command, os.environ)
    raise AssertionError("exec unexpectedly returned")


if __name__ == "__main__":
    sys.exit(main())
