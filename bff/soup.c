/* BFF primordial soup — C core.
 *
 * A straight transcription of bff.py's evaluate() plus the epoch loop, kept
 * in C only because the Python version is ~100x too slow to reach the phase
 * transition. soup.py checks the two against each other on startup.
 *
 * Build:  gcc -O2 -shared -fPIC -o libsoup.so soup.c
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define TAPE 64
#define MEM (2 * TAPE)
#define MASK (MEM - 1)

/* ---- PCG-ish 64-bit RNG, so runs are reproducible from a seed ---- */
static uint64_t rng_state = 0x853c49e6748fea9bULL;

static inline uint64_t rnd64(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

void seed_rng(uint64_t s) { rng_state = s ? s : 0x853c49e6748fea9bULL; }

/* ---- the interpreter ---- *
 * Returns the number of non-noop operations executed. `tape` is 128 bytes
 * and is modified in place.
 */
size_t evaluate(uint8_t *tape, size_t stepcount) {
    int pc = 0, head0 = 0, head1 = 0;
    size_t nskip = 0, i = 0;

    while (i < stepcount) {
        head0 &= MASK;
        head1 &= MASK;

        uint8_t cmd = tape[pc];
        int did = 1;

        switch (cmd) {
        case '<': head0--; break;
        case '>': head0++; break;
        case '{': head1--; break;
        case '}': head1++; break;
        case '+': tape[head0]++; break;
        case '-': tape[head0]--; break;
        case '.': tape[head1] = tape[head0]; break;
        case ',': tape[head0] = tape[head1]; break;
        case '[':
            if (tape[head0] == 0) {
                int depth = 1;
                pc++;
                while (pc < MEM && depth > 0) {
                    if (tape[pc] == ']') depth--;
                    else if (tape[pc] == '[') depth++;
                    pc++;
                }
                pc--;
                if (depth != 0) pc = MEM;
            }
            break;
        case ']':
            if (tape[head0] != 0) {
                int depth = 1;
                pc--;
                while (pc >= 0 && depth > 0) {
                    if (tape[pc] == ']') depth++;
                    else if (tape[pc] == '[') depth--;
                    pc--;
                }
                pc++;
                if (depth != 0) pc = -1;
            }
            break;
        default: did = 0; break;
        }

        if (!did) nskip++;
        i++;
        if (pc < 0) break;
        pc++;
        if (pc >= MEM) break;
    }
    return i - nskip;
}

/* ---- one epoch over the whole soup ----
 *
 * Shuffle the programs, pair them off, run split(exec(AB)) on each pair,
 * then apply per-byte background mutation.
 *
 * soup:        n_programs * 64 bytes
 * idx:         scratch buffer of n_programs uint32_t
 * mut_num:     mutation probability numerator, denominator is 1<<30
 * Returns total ops executed this epoch (the paper's "computation" measure).
 */
uint64_t run_epoch(uint8_t *soup, uint32_t n_programs, uint32_t *idx,
                   uint32_t mut_num, size_t stepcount) {
    /* Fisher-Yates */
    for (uint32_t i = 0; i < n_programs; i++) idx[i] = i;
    for (uint32_t i = n_programs - 1; i > 0; i--) {
        uint32_t j = (uint32_t)(rnd64() % (i + 1));
        uint32_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }

    uint64_t total_ops = 0;
    uint8_t tape[MEM];

    for (uint32_t p = 0; p + 1 < n_programs; p += 2) {
        uint8_t *a = soup + (size_t)idx[p] * TAPE;
        uint8_t *b = soup + (size_t)idx[p + 1] * TAPE;
        memcpy(tape, a, TAPE);
        memcpy(tape + TAPE, b, TAPE);
        total_ops += evaluate(tape, stepcount);
        memcpy(a, tape, TAPE);
        memcpy(b, tape + TAPE, TAPE);
    }

    if (mut_num > 0) {
        size_t total = (size_t)n_programs * TAPE;
        for (size_t k = 0; k < total; k++) {
            if ((rnd64() >> 34) < mut_num) soup[k] = (uint8_t)(rnd64() >> 40);
        }
    }
    return total_ops;
}

/* Fill the soup with uniformly random bytes. */
void randomize(uint8_t *soup, size_t nbytes) {
    for (size_t i = 0; i < nbytes; i++) soup[i] = (uint8_t)(rnd64() >> 40);
}