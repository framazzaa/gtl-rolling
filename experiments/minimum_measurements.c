// Exhaustive search over the single-qubit Pauli measurement patterns of a GTL graph state.
//
// Usage: minimum_measurements n_o kappa_b part parts
//
// Every qubit is left unmeasured (I) or measured in the X, Y or Z basis. The search covers the orchestration choices
// part, part + parts, ... out of 4^n_o, so that `parts` processes share the work, and prints
//   patterns <number of classes evaluated>
//   pairs_pure <largest number of decoupled Bell pairs for m = 0, ..., n, every orchestration qubit measured>
//   pairs_any <largest number of decoupled Bell pairs for m = 0, ..., n>
//   distribution <m> <pairs> <number of patterns>
// with m the number of measured qubits. Adjacencies are 32-bit masks, so n <= 32 and, with n_o >= 2, kappa_b <= 10.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint32_t u32;
enum { I, X, Y, Z };

static int n, n_o, kappa_b, kappa;
static int choice[32];
static long long classes;
static int pairs_pure[33], pairs_any[33];
static unsigned long long distribution[33][17], factorial[11];

// After each measurement the graph is the one of the corrected state. The local Clifford corrections change the basis
// in which a later measurement of a qubit acts on that graph, which `frame` tracks.
//
// Change of basis of the neighbors of a qubit measured in Y, whose correction exchanges X and Y.
static const unsigned char swap_y[4] = {I, Y, X, Z};
// Change of basis of the support of a qubit measured in X, whose correction exchanges X and Z.
static const unsigned char swap_x[4] = {I, Z, Y, X};

// Complements the neighborhood of v: toggles every edge between two neighbors of v.
static void local_complement(u32* adj, int v) {
  u32 nv = adj[v], m = nv;
  while (m) {
    int u = __builtin_ctz(m);
    m &= m - 1;
    adj[u] ^= nv & ~(1u << u);
  }
}

// Removes every edge of v.
static void remove_vertex(u32* adj, int v) {
  u32 m = adj[v];
  while (m) {
    int u = __builtin_ctz(m);
    m &= m - 1;
    adj[u] &= ~(1u << v);
  }
  adj[v] = 0;
}

// Measures v in `basis`, as seen through its frame, with the graph rule of the measurement that acts on the graph.
// The Pauli-X rule takes the smallest neighbor as support; an isolated qubit measured in X leaves the graph unchanged.
static void measure(u32* adj, unsigned char (*frame)[4], int v, int basis) {
  int q = frame[v][basis];
  if (q == Z) {
    remove_vertex(adj, v);
  } else if (q == Y) {
    u32 m = adj[v];
    local_complement(adj, v);
    remove_vertex(adj, v);
    while (m) {
      int u = __builtin_ctz(m);
      m &= m - 1;
      for (int k = X; k <= Z; k++) frame[u][k] = swap_y[frame[u][k]];
    }
  } else if (adj[v]) {
    int b0 = __builtin_ctz(adj[v]);
    local_complement(adj, b0);
    local_complement(adj, v);
    remove_vertex(adj, v);
    local_complement(adj, b0);
    for (int k = X; k <= Z; k++) frame[b0][k] = swap_x[frame[b0][k]];
  }
}

// Records a complete pattern with m measurements: counts its decoupled Bell pairs, i.e., the connected components of
// two unmeasured qubits, and adds the number of patterns of its class to the distribution.
static void record_pattern(const u32* adj, int m) {
  u32 unmeasured = 0, seen = 0;
  int pairs = 0, all_orchestration = 1;
  classes++;
  for (int v = 0; v < n; v++)
    if (choice[v] == I) unmeasured |= 1u << v;
  for (int i = 0; i < n_o; i++)
    if (choice[kappa + i] == I) all_orchestration = 0;
  for (int v = 0; v < n; v++) {
    if (!(unmeasured >> v & 1) || (seen >> v & 1)) continue;
    u32 component = 1u << v, todo = 1u << v;
    while (todo) {
      int u = __builtin_ctz(todo);
      todo &= todo - 1;
      u32 found = adj[u] & unmeasured & ~component;
      component |= found;
      todo |= found;
    }
    seen |= component;
    pairs += __builtin_popcount(component) == 2;
  }
  unsigned long long weight = 1;
  for (int g = 0; g <= n_o; g++) {
    int count[4] = {0};
    for (int v = g * kappa_b; v < (g + 1) * kappa_b; v++) count[choice[v]]++;
    weight *=
        factorial[kappa_b] / (factorial[count[I]] * factorial[count[X]] * factorial[count[Y]] * factorial[count[Z]]);
  }
  distribution[m][pairs] += weight;
  if (pairs > pairs_any[m]) pairs_any[m] = pairs;
  if (all_orchestration && pairs > pairs_pure[m]) pairs_pure[m] = pairs;
}

// Enumerates the choices of the peers v, v + 1, ..., kappa - 1, nondecreasing within each group, and records every
// complete pattern; m is the number of measurements so far. The peers of a group are twins, so the patterns that
// differ by a permutation of the peers of a group form a class: one pattern is evaluated per class, and weighted by the
// number of patterns of its class.
static void assign_peers(int v, const u32* adj, unsigned char (*frame)[4], int m) {
  if (v == kappa) {
    record_pattern(adj, m);
    return;
  }
  int lowest = v % kappa_b ? choice[v - 1] : I;
  for (int c = lowest; c <= Z; c++) {
    choice[v] = c;
    if (c == I) {
      assign_peers(v + 1, adj, frame, m);
    } else {
      u32 a[32];
      unsigned char f[32][4];
      memcpy(a, adj, sizeof(u32) * n);
      memcpy(f, frame, 4 * n);
      measure(a, f, v, c);
      assign_peers(v + 1, a, f, m + 1);
    }
  }
  choice[v] = I;
}

// Prints `name` followed by the values a[0], ..., a[n].
static void print(const char* name, const int* a) {
  printf("%s", name);
  for (int m = 0; m <= n; m++) printf(" %d", a[m]);
  printf("\n");
}

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s n_o kappa_b part parts\n", argv[0]);
    return 1;
  }
  n_o = atoi(argv[1]);
  kappa_b = atoi(argv[2]);
  int part = atoi(argv[3]), parts = atoi(argv[4]);
  kappa = (n_o + 1) * kappa_b;
  n = kappa + n_o;
  if (n_o < 2 || kappa_b < 1 || n > 32 || parts < 1 || part < 0 || part >= parts) {
    fprintf(stderr, "need n_o >= 2, kappa_b >= 1, at most 32 qubits and 0 <= part < parts\n");
    return 1;
  }
  factorial[0] = 1;
  for (int k = 1; k <= 10; k++) factorial[k] = factorial[k - 1] * k;
  u32 adj[32] = {0};
  for (int i = 0; i < n_o; i++)
    for (int c = i * kappa_b; c < (i + 2) * kappa_b; c++) {
      adj[kappa + i] |= 1u << c;
      adj[c] |= 1u << (kappa + i);
    }
  unsigned char frame[32][4];
  for (int u = 0; u < n; u++)
    for (int k = I; k <= Z; k++) frame[u][k] = k;

  long long tier2 = 1;
  for (int i = 0; i < n_o; i++) tier2 *= 4;
  for (long long t = part; t < tier2; t += parts) {
    u32 a[32];
    unsigned char f[32][4];
    memcpy(a, adj, sizeof(u32) * n);
    memcpy(f, frame, 4 * n);
    int m = 0;
    long long r = t;
    for (int i = 0; i < n_o; i++, r /= 4) {
      choice[kappa + i] = r % 4;
      if (r % 4) {
        measure(a, f, kappa + i, r % 4);
        m++;
      }
    }
    assign_peers(0, a, f, m);
  }
  printf("patterns %lld\n", classes);
  print("pairs_pure", pairs_pure);
  print("pairs_any", pairs_any);
  for (int m = 0; m <= n; m++)
    for (int p = 0; p <= n / 2; p++)
      if (distribution[m][p]) printf("distribution %d %d %llu\n", m, p, distribution[m][p]);
  return 0;
}
