#ifndef GNN_TYPES_H
#define GNN_TYPES_H

#include <ap_int.h>
#include <hls_stream.h>

// Override these via Makefile -D flags for fast micro-benchmarking
#ifndef MAX_N
#define MAX_N 256
#endif

#ifndef MAX_K
#define MAX_K 9
#endif

#ifndef MAX_D
#define MAX_D 64
#endif

#define MAX_H 8
#define D_H (MAX_D / MAX_H)

// Struct for distance sorting
struct DistanceValue {
    float dist;
    int index;
    bool operator<(const DistanceValue& other) const { return dist < other.dist; }
};

#endif
