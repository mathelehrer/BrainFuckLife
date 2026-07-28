#!/usr/bin/env python3
"""Brainfuck interpreter.

Usage:
    python3 bf.py program.bf
    python3 bf.py -c '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
"""
import sys


def run(code: str, input_data: str | None = None) -> None:
    jump = {}
    stack = []
    for i, c in enumerate(code):
        if c == "[":
            stack.append(i)
        elif c == "]":
            if not stack:
                raise SyntaxError(f"unmatched ']' at position {i}")
            j = stack.pop()
            jump[i] = j
            jump[j] = i
    if stack:
        raise SyntaxError(f"unmatched '[' at position {stack[-1]}")

    if input_data is None and "," in code:
        input_data = sys.stdin.read()
    input_data = input_data or ""

    tape = bytearray(30000)
    ptr = 0
    ip = 0
    in_pos = 0
    out = []

    while ip < len(code):
        c = code[ip]
        if c == ">":
            ptr = (ptr + 1) % len(tape)
        elif c == "<":
            ptr = (ptr - 1) % len(tape)
        elif c == "+":
            tape[ptr] = (tape[ptr] + 1) % 256
        elif c == "-":
            tape[ptr] = (tape[ptr] - 1) % 256
        elif c == ".":
            out.append(chr(tape[ptr]))
        elif c == ",":
            if in_pos < len(input_data):
                tape[ptr] = ord(input_data[in_pos])
                in_pos += 1
            else:
                tape[ptr] = 0
        elif c == "[":
            if tape[ptr] == 0:
                ip = jump[ip]
        elif c == "]":
            if tape[ptr] != 0:
                ip = jump[ip]
        ip += 1

    sys.stdout.write("".join(out))


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    if args[0] == "-c":
        code = args[1]
    else:
        with open(args[0]) as f:
            code = f.read()

    run(code)


if __name__ == "__main__":
    main()
