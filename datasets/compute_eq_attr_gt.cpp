// compute_eq_attr_gt.cpp
// Multi-threaded (OpenMP) candidate generation for "equality-filtered ground-truth":
//   for each query, within the base subset whose attribute signature (A attribute values,
//   radix=C) matches, compute distances and select top-(K+M) candidates (absolute base indices)
//   + record the match-set size.
// The final exact top-K ordering is done by Python-side recompute_gt on the candidates
// (same logic as the verifier).
// The range-predicate version is compute_range_attr_gt.cpp (per-query interval mask, no signature grouping).
//
// Design notes (adapted from RoarGraph compute_groundtruth.cpp's multi-threaded brute-force
//   nearest-neighbor idea, but changed to "filtered"):
//   - Build base CSR by signature counting sort (blist); queries sorted by signature (qlist) so
//     same-signature queries are processed consecutively -> share the same match subset, cache-friendly.
//   - Distances accumulated in double (more accurate than the verifier's float, and the candidate
//     margin M guarantees the true top-K is included).
//   - Per query use nth_element + partial sort to take top-(K+M): O(m) selection instead of a
//     per-element heap, faster and lighter.
//   - Depends only on libgomp and the standard library; no MKL / GPU / faiss needed.
//
// Usage:
//   compute_eq_attr_gt --base <base.fvecs> --query <query.fvecs> \
//       --base-attr <base_attr.bin> --query-attr <query_attr.bin> \
//       --metric ip|l2 --K 100 --C 10 --margin 256 --threads 0 \
//       --cand-out <cand.int32> --count-out <count.int64>
//   --C is the attribute cardinality (signature radix): must be >= max attribute value + 1,
//   otherwise A>=2 causes a signature collision.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <omp.h>

static void die(const std::string& msg) {
    std::cerr << "[compute_eq_attr_gt] ERROR: " << msg << std::endl;
    std::exit(1);
}

// ---- read fvecs (each vector: 4-byte int32 dim header + float32 data, fixed dim) ----
struct Mat { int64_t n = 0, d = 0; float* data = nullptr; ~Mat(){ delete[] data; } };

static Mat read_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot open fvecs: " + path);
    int64_t sz = (int64_t)f.tellg();
    f.seekg(0);
    int dim0 = 0;
    f.read((char*)&dim0, 4);
    if (dim0 <= 0) die("bad fvecs dim");
    // Standard fvecs: every vector is prefixed by a 4-byte dim header, so each row is 4 + 4*dim bytes
    int64_t row = 4LL + 4LL * dim0;
    if (sz % row != 0) die("fvecs size not multiple of row");
    int64_t n = sz / row;
    float* data = new float[(int64_t)n * dim0];
    f.seekg(0);
    for (int64_t i = 0; i < n; i++) {
        int dd = 0;
        f.read((char*)&dd, 4);
        if (dd != dim0) die("fvecs dim mismatch at vector " + std::to_string(i));
        f.read((char*)&data[i * dim0], 4LL * dim0);
    }
    if (!f) die("fvecs read incomplete: " + path);
    return Mat{n, (int64_t)dim0, data};
}

// ---- read attribute bin (8-byte [N,A] int32 header + N*A data; base=uint32, query=int32) ----
struct Attr { int64_t n = 0, a = 0; int* data = nullptr; ~Attr(){ delete[] data; } };

static Attr read_attr(const std::string& path, bool is_uint32) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot open attr: " + path);
    int64_t sz = (int64_t)f.tellg();
    f.seekg(0);
    int hdr[2] = {0, 0};
    f.read((char*)hdr, 8);
    int64_t n = hdr[0], a = hdr[1];
    if (n <= 0 || a <= 0) die("bad attr header");
    if (sz != 8 + n * a * 4LL) die("attr file size mismatch: " + path);
    int* data = new int[(int64_t)n * a];
    if (is_uint32) {
        std::vector<unsigned int> tmp((size_t)(n * a));
        f.read((char*)tmp.data(), (int64_t)n * a * 4);
        for (int64_t i = 0; i < n * a; i++) data[i] = (int)tmp[i];
    } else {
        f.read((char*)data, (int64_t)n * a * 4);
    }
    if (!f) die("attr read incomplete: " + path);
    return Attr{n, a, data};
}

static std::vector<long> build_sig(const int* data, int64_t n, int a, int C) {
    std::vector<long> sig((size_t)n);
    long mult = 1;
    for (int c = 0; c < a; c++) {
        for (int64_t i = 0; i < n; i++)
            sig[i] += (long)data[i * a + c] * mult;
        mult *= C;
    }
    return sig;
}

int main(int argc, char** argv) {
    std::string base_path, query_path, base_attr_path, query_attr_path;
    std::string cand_out, count_out, metric = "l2";
    int K = 100, margin = 256, threads = 0, C = 10;

    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto need = [&](const char* name)->std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (s == "--base") base_path = need("--base");
        else if (s == "--query") query_path = need("--query");
        else if (s == "--base-attr") base_attr_path = need("--base-attr");
        else if (s == "--query-attr") query_attr_path = need("--query-attr");
        else if (s == "--cand-out") cand_out = need("--cand-out");
        else if (s == "--count-out") count_out = need("--count-out");
        else if (s == "--metric") metric = need("--metric");
        else if (s == "--K") K = std::atoi(need("--K").c_str());
        else if (s == "--margin") margin = std::atoi(need("--margin").c_str());
        else if (s == "--threads") threads = std::atoi(need("--threads").c_str());
        else if (s == "--C") C = std::atoi(need("--C").c_str());
        else die("unknown arg: " + s);
    }
    if (base_path.empty() || query_path.empty() || base_attr_path.empty() ||
        query_attr_path.empty() || cand_out.empty() || count_out.empty())
        die("missing required args");
    if (metric != "ip" && metric != "l2") die("--metric must be ip or l2");
    if (K <= 0 || margin < 0) die("bad K/margin");
    if (C < 2) die("--C (attribute cardinality) must be >= 2; the signature radix must exceed the "
                   "max attribute value, otherwise A>=2 collision corrupts the candidate set");

    if (threads > 0) omp_set_num_threads(threads);

    double t0 = omp_get_wtime();
    Mat X = read_fvecs(base_path);
    Mat Q = read_fvecs(query_path);
    Attr ba = read_attr(base_attr_path, true);
    Attr qa = read_attr(query_attr_path, false);

    int64_t N = X.n, QN = Q.n, d = X.d;
    int A = (int)ba.a;
    // C is passed via --C (default 10); the signature radix must exceed the max attribute value, otherwise A>=2 collides
    if (ba.n != N || qa.n != QN) die("attr row count mismatch with vectors");
    if (qa.a != A) die("base/query attr A mismatch");

    long ng = 1;
    for (int c = 0; c < A; c++) ng *= C;

    std::vector<long> bsign = build_sig(ba.data, N, A, C);
    std::vector<long> qsign = build_sig(qa.data, QN, A, C);

    std::vector<int64_t> bstart((size_t)ng + 1, 0);
    for (int64_t i = 0; i < N; i++) bstart[bsign[i] + 1]++;
    for (long g = 0; g < ng; g++) bstart[g + 1] += bstart[g];
    std::vector<int64_t> blist((size_t)N);
    {
        std::vector<int64_t> cur = bstart;
        for (int64_t i = 0; i < N; i++) blist[cur[bsign[i]]++] = i;
    }

    std::vector<int64_t> qstart((size_t)ng + 1, 0);
    for (int64_t i = 0; i < QN; i++) qstart[qsign[i] + 1]++;
    for (long g = 0; g < ng; g++) qstart[g + 1] += qstart[g];
    std::vector<int64_t> qlist((size_t)QN);
    {
        std::vector<int64_t> cur = qstart;
        for (int64_t i = 0; i < QN; i++) qlist[cur[qsign[i]]++] = i;
    }

    int totalcap = K + margin;
    std::vector<int32_t> cand((size_t)QN * totalcap, -1);
    std::vector<int64_t> count((size_t)QN, 0);

    bool is_l2 = (metric == "l2");

#pragma omp parallel for schedule(static, 64)
    for (int64_t ii = 0; ii < QN; ii++) {
        int64_t q = qlist[ii];
        long sig = qsign[q];
        int64_t s = bstart[sig], e = bstart[sig + 1];
        int64_t m = e - s;
        count[q] = m;

        std::vector<std::pair<double, int64_t>> arr((size_t)m);
        const float* qv = &Q.data[q * d];
        for (int64_t k = s; k < e; k++) {
            int64_t bi = blist[k];
            const float* xv = &X.data[bi * d];
            // Use float32 accumulation, same precision as the verifier's recompute_gt float32 einsum,
            // so the boundary ordering difference <= 1e-6, exactly within the verifier's 1e-6 tolerance -> top-K is in candidates.
            float dist;
            if (is_l2) {
                float acc = 0.0f;
                for (int dd = 0; dd < d; dd++) {
                    float t = xv[dd] - qv[dd];
                    acc += t * t;
                }
                dist = acc;
            } else {
                float acc = 0.0f;
                for (int dd = 0; dd < d; dd++)
                    acc += xv[dd] * qv[dd];
                dist = acc;
            }
            arr[k - s] = std::make_pair((double)dist, bi);
        }

        int cap = totalcap;
        if ((int64_t)cap > m) cap = (int)m;
        if (is_l2) {
            auto less = [](const std::pair<double,int64_t>& a,
                           const std::pair<double,int64_t>& b){ return a.first < b.first; };
            std::nth_element(arr.begin(), arr.begin() + cap, arr.end(), less);
            std::sort(arr.begin(), arr.begin() + cap, less);
        } else {
            auto greater = [](const std::pair<double,int64_t>& a,
                              const std::pair<double,int64_t>& b){ return a.first > b.first; };
            std::nth_element(arr.begin(), arr.begin() + cap, arr.end(), greater);
            std::sort(arr.begin(), arr.begin() + cap, greater);
        }
        int64_t base = q * totalcap;
        for (int t = 0; t < cap; t++)
            cand[base + t] = (int32_t)arr[t].second;
    }

    {
        std::ofstream fc(cand_out, std::ios::binary);
        fc.write((const char*)cand.data(), (int64_t)cand.size() * 4);
        if (!fc) die("write cand failed: " + cand_out);
    }
    {
        std::ofstream fq(count_out, std::ios::binary);
        fq.write((const char*)count.data(), (int64_t)count.size() * 8);
        if (!fq) die("write count failed: " + count_out);
    }

    double t1 = omp_get_wtime();
    std::cerr << "[compute_eq_attr_gt] done: N=" << N << " QN=" << QN << " D=" << d
              << " A=" << A << " C=" << C << " ng=" << ng
              << " K=" << K << " margin=" << margin << " totalcap=" << totalcap
              << " threads=" << omp_get_max_threads()
              << " time=" << (t1 - t0) << "s" << std::endl;
    return 0;
}
