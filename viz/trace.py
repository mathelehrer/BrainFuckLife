#!/usr/bin/env python3
"""Trace a Brainfuck program's execution for visualization.

Records one "step" per command, except consecutive runs of the same
+/- are collapsed into a single step (matching how a human would read
"++++++++" as "add 8", not eight separate increments).
"""
import json
import sys


def trace(code: str) -> dict:
    jump = {}
    stack = []
    for i, c in enumerate(code):
        if c == "[":
            stack.append(i)
        elif c == "]":
            j = stack.pop()
            jump[i] = j
            jump[j] = i

    tape = bytearray(30000)
    ptr = 0
    ip = 0
    steps = []  # list of {op, ptr, loop_depth, cells: {idx: val}, output}
    loop_depth = 0
    max_cell = 0

    while ip < len(code):
        c = code[ip]

        if c in "+-":
            start = ip
            run = c
            while ip + 1 < len(code) and code[ip + 1] == c:
                ip += 1
                run += c
            n = len(run)
            delta = n if c == "+" else -n
            tape[ptr] = (tape[ptr] + delta) % 256
            steps.append({
                "op": f"{'+' if c == '+' else '-'}{n}",
                "ptr": ptr,
                "loop_depth": loop_depth,
                "cell": ptr,
                "value": tape[ptr],
                "output": None,
            })
            max_cell = max(max_cell, ptr)
        elif c == ">":
            ptr = (ptr + 1) % len(tape)
            max_cell = max(max_cell, ptr)
            steps.append({
                "op": ">", "ptr": ptr, "loop_depth": loop_depth,
                "cell": ptr, "value": tape[ptr], "output": None,
            })
        elif c == "<":
            ptr = (ptr - 1) % len(tape)
            steps.append({
                "op": "<", "ptr": ptr, "loop_depth": loop_depth,
                "cell": ptr, "value": tape[ptr], "output": None,
            })
        elif c == ".":
            ch = chr(tape[ptr])
            steps.append({
                "op": ".", "ptr": ptr, "loop_depth": loop_depth,
                "cell": ptr, "value": tape[ptr], "output": ch,
            })
        elif c == ",":
            steps.append({
                "op": ",", "ptr": ptr, "loop_depth": loop_depth,
                "cell": ptr, "value": tape[ptr], "output": None,
            })
        elif c == "[":
            if tape[ptr] == 0:
                ip = jump[ip]
                steps.append({
                    "op": "[ (skip)", "ptr": ptr, "loop_depth": loop_depth,
                    "cell": ptr, "value": tape[ptr], "output": None,
                })
            else:
                loop_depth += 1
                steps.append({
                    "op": "[ (enter)", "ptr": ptr, "loop_depth": loop_depth,
                    "cell": ptr, "value": tape[ptr], "output": None,
                })
        elif c == "]":
            if tape[ptr] != 0:
                steps.append({
                    "op": "] (loop)", "ptr": ptr, "loop_depth": loop_depth,
                    "cell": ptr, "value": tape[ptr], "output": None,
                })
                ip = jump[ip]
            else:
                steps.append({
                    "op": "] (exit)", "ptr": ptr, "loop_depth": loop_depth,
                    "cell": ptr, "value": tape[ptr], "output": None,
                })
                loop_depth -= 1
        ip += 1

    n_cells = max_cell + 1
    # Build a per-step, per-cell value matrix with carry-forward.
    cur = [0] * n_cells
    matrix = []
    for s in steps:
        cur[s["cell"]] = s["value"]
        matrix.append(list(cur))

    return {
        "n_cells": n_cells,
        "steps": steps,
        "matrix": matrix,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/hello.bf"
    with open(path) as f:
        code = f.read()
    result = trace(code)
    print(f"steps: {len(result['steps'])}  cells used: {result['n_cells']}", file=sys.stderr)
    print(json.dumps(result))
