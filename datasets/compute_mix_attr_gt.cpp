// compute_mix_attr_gt.cpp
// Multi-threaded (OpenMP) candidate generation for "mixed-filtered ground-truth":
//   each vector carries two attributes: one equality attribute + one range attribute
//   (base attr (N,2): column 0 = equality category in [0,C_eq), column 1 = range category in [0,C_range)).
//   For each query (query attr (QN,3): columns = [eq_val, lo, hi]), the match subset is the union
//   where base equality attribute == eq_val **AND** range attribute falls in [lo, hi];
//   within the match subset compute distances and select top-(K+M) candidates (absolute base indices)
//   + record the match-set size.
// The final exact top-K ordering is done by Python-side code on the candidates (same as self-check).
//
// Selectivity: sel = (1/C_eq) * (w / C_range), w = hi-lo+1 (default recipe C_eq=10, C_range=100, w=10 -> 1%).
//
// The only difference from compute_range_attr_gt.cpp (pure range) is the mask condition: a range
//   predicate cannot be pre-grouped by signature; this program also applies a per-query two-condition
//   mask over the N rows and collects matching indices, then computes distances within the match subset.
//   Distances accumulated in double: for L2 distances of large magnitude with densely packed distances
//   in the match set, float32 accumulation's ULP is enough to swap hundreds of ranks at the boundary,
//   causing the margin window to miss the true top-K; with double accumulation the candidate window
//   precisely contains the true top-K, and Python finalize/self-check still use float32, agreeing within 1e-6.
//
// Depends only on libgomp and the standard library; no MKL / GPU / faiss needed.
//
// Usage:
//   compute_mix_attr_gt --base <base.fvecs> --query <query.fvecs> \
//       --base-attr <base_attr.bin> --query-attr <query_attr_mix.bin> \
//       --metric ip|l2 --K 100 --margin 256 --threads 0 \
//       --cand-out <cand.int32> --count-out <count.int64>
//   File layout: base-attr = 8-byte [N,2] int32 header + N*2 uint32;
//                query-attr = 8-byte [QN,3] int32 header + QN*3 int32 [eq_val,lo,hi].
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
    std::cerr << "[compute_mix_attr_gt] ERROR: " << msg << std::endl;
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

int main(int argc, char** argv) {
    std::string base_path, query_path, base_attr_path, query_attr_path;
    std::string cand_out, count_out, metric = "l2";
    int K = 100, margin = 256, threads = 0;

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
        else die("unknown arg: " + s);
    }
    if (base_path.empty() || query_path.empty() || base_attr_path.empty() ||
        query_attr_path.empty() || cand_out.empty() || count_out.empty())
        die("missing required args");
    if (metric != "ip" && metric != "l2") die("--metric must be ip or l2");
    if (K <= 0 || margin < 0) die("bad K/margin");

    if (threads >0) omp_set_num_threads(threads);

    double t0 = omp_get_wtime();
    Mat X = read_fvecs(base_path);
    Mat Q = read_fvecs(query_path);
    Attr ba = read_attr(base_attr_path, true);    // base mixed attr (N,2) uint32: [eq_cat, range_cat]
    Attr qa = read_attr(query_attr_path, false);  // query mixed attr (QN,3) int32: [eq_val, lo, hi]

    int64_t N = X.n, QN = Q.n, d = X.d;
    if (ba.a != 2) die("mixed base attr must have 2 columns [eq_cat, range_cat]");
    if (qa.a != 3) die("mixed query attr must have 3 columns [eq_val, lo, hi]");
    if (ba.n != N || qa.n != QN) die("attr row count mismatch with vectors");

    int totalcap = K + margin;
    std::vector<int32_t> cand((size_t)QN * totalcap, -1);
    std::vector<int64_t> count((size_t)QN, 0);

    bool is_l2 = (metric == "l2");

#pragma omp parallel for schedule(dynamic, 4)
    for (int64_t q = 0; q < QN; q++) {
        // This query's equality target and range interval
        const int eqv = qa.data[(size_t)q * 3 + 0];
        const int lo  = qa.data[(size_t)q * 3 + 1];
        const int hi  = qa.data[(size_t)q * 3 + 2];

        // ---- mask: per base row check "equality match AND range-in-interval", collect matching indices ----
        std::vector<int64_t> mlist;
        mlist.reserve(1024);
        for (int64_t i = 0; i < N; i++) {
            const int* bp = &ba.data[(size_t)i * 2];
            if (bp[0] == eqv && bp[1] >= lo && bp[1] <= hi) mlist.push_back(i);
        }
        int64_t m = (int64_t)mlist.size();
        count[q] = m;

        int cap = totalcap;
        if ((int64_t)cap > m) cap = (int)m;
        if (m == 0) continue;

        // ---- within the match subset, compute distances and take top-(K+M) ----
        // Distances accumulated in double (same precision rationale as compute_range_attr_gt.cpp):
        // the candidate window precisely contains the true top-K; Python finalize/self-check use
        // float32, agreeing within the 1e-6 tolerance.
        std::vector<std::pair<double, int64_t>> arr((size_t)m);
        const float* qv = &Q.data[q * d];
        for (int64_t k = 0; k < m; k++) {
            int64_t bi = mlist[(size_t)k];
            const float* xv = &X.data[bi * d];
            double dist;
            if (is_l2) {
                double acc = 0.0;
                for (int dd = 0; dd < d; dd++) {
                    double t = (double)xv[dd] - (double)qv[dd];
                    acc += t * t;
                }
                dist = acc;
            } else {
                double acc = 0.0;
                for (int dd = 0; dd < d; dd++)
                    acc += (double)xv[dd] * (double)qv[dd];
                dist = acc;
            }
            arr[(size_t)k] = std::make_pair(dist, bi);
        }

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
            cand[(size_t)base + t] = (int32_t)arr[(size_t)t].second;
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
    std::cerr << "[compute_mix_attr_gt] done: N=" << N << " QN=" << QN << " D=" << d
              << " totalcap=" << totalcap
              << " threads=" << omp_get_max_threads()
              << " time=" << (t1 - t0) << "s" << std::endl;
    return 0;
}
