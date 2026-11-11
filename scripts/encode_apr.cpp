// ============================================================================
// encode_apr.cpp -- Attribute APR/CDMA phase-superposition encoder (C++ version,
// replacing encode_attr_apr.py).
//
// Purpose: apply an APR rotation encoding to the unified dataset (each dimension
// pair simultaneously carries the phase code of all A attributes; orthogonal
// rotation preserves distance, so the original-space GT and exact reordering stay
// valid). base/query receive the same salt-determined phases, hence they stay
// consistent and the GT remains valid.
//
// Relation to the Python encode_attr_apr.py / _dataset_utils.encode_apr:
//   - CLI is fully compatible (plus --salt-file / --threads); run.sh is unchanged.
//   - Phase hashing is bit-identical to _hash_to_phases_apr (splitmix64 finalizer
//     _mix64):
//         base = mix64(v * GOLDEN)
//         h    = mix64(base ^ salt[j]) & 0x7FFFFFFF
//         phase = (h / 2147483647.0) * 2*pi
//     salt comes from --salt-file (numpy-exact A*P int64 values), keeping
//     continuity with existing sift/msmarco/openai1536 APR results and
//     byte-identical reruns.
//   - Numeric path: float32 in -> double compute -> cast float32; not normalized
//     (consistent with Python, only a non-unit-norm warning is printed).
//
// Performance: precompute a "attribute tuple (v_0..v_{A-1}) -> total phase
// cos/sin table" (for this config C^A <= 1000), then each row does an O(1) table
// lookup and applies the rotation to all dimension pairs; streaming chunks +
// per-row multithreading (64MB/chunk), peak memory O(chunk + table), independent
// of file size.
//
// Build (consistent with the reference implementation):
//   g++ -std=c++17 -o ../bin/encode_apr encode_apr.cpp -I . \
//       -Ofast -march=core-avx2 -pthread -fno-fast-math -ffp-contract=off
// Note: -fno-fast-math is required (the pairwise norm sum depends on strict
// floating-point associativity, otherwise it diverges from numpy bit-exactly).
// ============================================================================
#include <sys/stat.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <getopt.h>

#include "apr_encode_io.hpp"

namespace {

using apr_io::die;
using apr_io::fvecs_probe;
using apr_io::attr_bin_read;
using apr_io::copy_file;
using apr_io::NormStat;

const uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
const double k2Pi = 2.0 * 3.14159265358979323846;

// Bit-identical to _dataset_utils._mix64 (splitmix64 finalizer).
inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Single phase from _hash_to_phases_apr: v is the non-negative attribute value,
// saltj is the uint64 bit pattern.
inline double phase_f0(uint64_t v, uint64_t saltj) {
    const uint64_t base = mix64(v * GOLDEN);
    const uint64_t h = mix64(base ^ saltj) & 0x7FFFFFFFULL;
    return (static_cast<double>(h) / 2147483647.0) * k2Pi;
}

struct Args {
    std::string ds, out;
    int d = 1536, A = 1, enc_seed = 1234;
    std::string base_name = "base_100k.fvecs";
    std::string query_name = "query_1k.fvecs";
    std::string base_attr_name = "base_attr.bin";
    std::string query_attr_name = "query_attr.bin";
    std::string groundtruth_name = "gt.ivecs";
    std::string apr_base_name = "apr_base.fvecs";
    std::string apr_query_name = "apr_query.fvecs";
    std::string salt_file;
    size_t threads = 0, chunk_rows = 0;
};

void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s --ds DIR --out DIR --d D --A A --salt-file F [options]\n"
                 "  --ds DIR              input unified dataset directory\n"
                 "  --out DIR             output APR directory\n"
                 "  --d D                 vector dimension\n"
                 "  --A A                 number of attributes\n"
                 "  --enc-seed S          encode seed (record only; salt comes from --salt-file)\n"
                 "  --base-name F         input base vector file name\n"
                 "  --query-name F        input query vector file name\n"
                 "  --base-attr-name F    input base attribute file name (uint32 self-describing bin)\n"
                 "  --query-attr-name F   input query attribute file name (int32 self-describing bin)\n"
                 "  --groundtruth-name F  input groundtruth file name (copied to out if present)\n"
                 "  --apr-base-name F     output APR base file name\n"
                 "  --apr-query-name F    output APR query file name\n"
                 "  --salt-file F         numpy-exact A*P int64 salt (required)\n"
                 "  --threads N           parallel thread count (default hardware_concurrency)\n"
                 "  --chunk-rows N        rows per chunk (default derived from 64MB/chunk)\n"
                 "  -h, --help            show help\n",
                 prog);
}

Args parse_args(int argc, char* argv[]) {
    Args a;
    static const struct option longopts[] = {
        {"ds", required_argument, nullptr, 1},
        {"out", required_argument, nullptr, 2},
        {"d", required_argument, nullptr, 3},
        {"A", required_argument, nullptr, 4},
        {"enc-seed", required_argument, nullptr, 5},
        {"base-name", required_argument, nullptr, 6},
        {"query-name", required_argument, nullptr, 7},
        {"base-attr-name", required_argument, nullptr, 8},
        {"query-attr-name", required_argument, nullptr, 9},
        {"groundtruth-name", required_argument, nullptr, 10},
        {"queries-name", required_argument, nullptr, 11},  // ignored: compat with Python version
        {"apr-base-name", required_argument, nullptr, 12},
        {"apr-query-name", required_argument, nullptr, 13},
        {"salt-file", required_argument, nullptr, 14},
        {"threads", required_argument, nullptr, 15},
        {"chunk-rows", required_argument, nullptr, 16},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int ind = 0, iarg = 0;
    while ((iarg = getopt_long(argc, argv, "", longopts, &ind)) != -1) {
        switch (iarg) {
            case 1: a.ds = optarg; break;
            case 2: a.out = optarg; break;
            case 3: a.d = std::atoi(optarg); break;
            case 4: a.A = std::atoi(optarg); break;
            case 5: a.enc_seed = std::atoi(optarg); break;
            case 6: a.base_name = optarg; break;
            case 7: a.query_name = optarg; break;
            case 8: a.base_attr_name = optarg; break;
            case 9: a.query_attr_name = optarg; break;
            case 10: a.groundtruth_name = optarg; break;
            case 11: break;  // queries-name ignored
            case 12: a.apr_base_name = optarg; break;
            case 13: a.apr_query_name = optarg; break;
            case 14: a.salt_file = optarg; break;
            case 15: a.threads = static_cast<size_t>(std::atoll(optarg)); break;
            case 16: a.chunk_rows = static_cast<size_t>(std::atoll(optarg)); break;
            case 'h': usage(argv[0]); std::exit(0);
            default: usage(argv[0]); std::exit(1);
        }
    }
    if (a.ds.empty()) die("missing --ds");
    if (a.out.empty()) die("missing --out");
    if (a.d < 2) die("--d must be >= 2 (rotation acts on dimension pairs), got %d", a.d);
    if (a.A < 1) die("--A must be >= 1, got %d", a.A);
    if (a.salt_file.empty()) die("missing --salt-file (must be numpy-exact)");
    if (a.threads == 0) a.threads = std::thread::hardware_concurrency();
    if (a.threads == 0) a.threads = 1;
    return a;
}

// Read salt: A*P int64 (little-endian), reinterpreted as uint64 bit patterns
// (consistent with numpy int64).
std::vector<uint64_t> read_salt(const std::string& path, size_t A, size_t P) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) die("cannot open salt file: %s", path.c_str());
    std::vector<int64_t> tmp(A * P);
    if (std::fread(tmp.data(), sizeof(int64_t), A * P, f) != A * P) {
        std::fclose(f);
        die("salt file too short (expected %zu int64): %s", A * P, path.c_str());
    }
    std::fclose(f);
    std::vector<uint64_t> salt(A * P);
    for (size_t i = 0; i < A * P; ++i) salt[i] = static_cast<uint64_t>(tmp[i]);
    return salt;
}

using TupleKey = std::vector<uint32_t>;

struct TableEntry {
    std::vector<double> c;  // length P
    std::vector<double> s;  // length P
};

// Single pass over the attribute matrix, building the "tuple -> total phase
// cos/sin table". Tuple = one row's attribute values (v_0..v_{A-1});
// total[j] = sum_a phase_f0(v_a, salt[a*P+j]).
std::map<TupleKey, TableEntry> build_table(const std::vector<int64_t>& attrs, size_t n,
                                           size_t A, size_t P,
                                           const std::vector<uint64_t>& salt) {
    std::map<TupleKey, TableEntry> tab;
    for (size_t i = 0; i < n; ++i) {
        TupleKey key(A);
        for (size_t a = 0; a < A; ++a) key[a] = static_cast<uint32_t>(attrs[i * A + a]);
        if (tab.find(key) != tab.end()) continue;
        TableEntry e;
        e.c.assign(P, 0.0);
        e.s.assign(P, 0.0);
        for (size_t j = 0; j < P; ++j) {
            double tot = 0.0;
            for (size_t a = 0; a < A; ++a) {
                const uint64_t v = static_cast<uint64_t>(attrs[i * A + a]);
                tot += phase_f0(v, salt[a * P + j]);
            }
            e.c[j] = std::cos(tot);
            e.s[j] = std::sin(tot);
        }
        tab.emplace(key, std::move(e));
    }
    return tab;
}

// Streaming chunks + per-row multithreading apply the APR rotation (table is
// read-only, lock-free); returns norm statistics.
NormStat encode_apr_file(const std::string& in_path, const std::string& out_path, int d,
                         size_t A, const std::vector<int64_t>& attrs, size_t n,
                         const std::map<TupleKey, TableEntry>& tab, size_t threads,
                         size_t chunk_rows) {
    const size_t D = static_cast<size_t>(d);
    const size_t P = D / 2;
    const size_t row_bytes = 4 + D * 4;
    FILE* fin = std::fopen(in_path.c_str(), "rb");
    if (fin == nullptr) die("cannot open input: %s", in_path.c_str());
    FILE* fout = std::fopen(out_path.c_str(), "wb");
    if (fout == nullptr) die("cannot open output: %s", out_path.c_str());
    std::vector<char> buf(chunk_rows * row_bytes);
    std::vector<char> obuf(1 << 20);
    std::setvbuf(fout, obuf.data(), _IOFBF, obuf.size());

    NormStat total{};
    for (size_t r0 = 0; r0 < n; r0 += chunk_rows) {
        const size_t rows = (n - r0 < chunk_rows) ? (n - r0) : chunk_rows;
        if (std::fread(buf.data(), row_bytes, rows, fin) != rows) {
            die("read failed: %s (at row %zu)", in_path.c_str(), r0);
        }
        for (size_t i = 0; i < rows; ++i) {
            int32_t h = 0;
            std::memcpy(&h, buf.data() + i * row_bytes, 4);
            if (h != d) die("fvecs row %zu dim header=%d expected %d: %s", r0 + i, h, d, in_path.c_str());
        }
        std::vector<NormStat> stats(threads);
        std::vector<std::thread> ts;
        ts.reserve(threads);
        const size_t per = (rows + threads - 1) / threads;
        for (size_t t = 0; t < threads; ++t) {
            const size_t loc0 = t * per;
            if (loc0 >= rows) break;
            const size_t loc1 = (loc0 + per < rows) ? (loc0 + per) : rows;
            ts.emplace_back([&, loc0, loc1, t]() {
                double s1 = 0.0, s2 = 0.0;
                for (size_t i = loc0; i < loc1; ++i) {
                    float* v = reinterpret_cast<float*>(buf.data() + i * row_bytes + 4);
                    TupleKey key(A);
                    const int64_t* ar = attrs.data() + (r0 + i) * A;
                    for (size_t a = 0; a < A; ++a) key[a] = static_cast<uint32_t>(ar[a]);
                    auto it = tab.find(key);
                    if (it == tab.end()) die("tuple not found (table build missed it): row %zu", r0 + i);
                    const TableEntry& e = it->second;
                    for (size_t j = 0; j < P; ++j) {
                        const double c = e.c[j], s = e.s[j];
                        const double x0 = v[2 * j], x1 = v[2 * j + 1];
                        v[2 * j] = static_cast<float>(x0 * c - x1 * s);
                        v[2 * j + 1] = static_cast<float>(x0 * s + x1 * c);
                    }
                    double n2 = 0.0;
                    for (size_t j = 0; j < D; ++j) {
                        const double x = v[j];
                        n2 += x * x;
                    }
                    const double nrm = std::sqrt(n2);
                    s1 += nrm;
                    s2 += nrm * nrm;
                }
                stats[t].sum = s1;
                stats[t].sum2 = s2;
                stats[t].count = loc1 - loc0;
            });
        }
        for (auto& th : ts) th.join();
        if (std::fwrite(buf.data(), row_bytes, rows, fout) != rows) {
            die("write failed: %s", out_path.c_str());
        }
        for (const auto& s : stats) {
            total.sum += s.sum;
            total.sum2 += s.sum2;
            total.count += s.count;
        }
    }
    std::fclose(fin);
    if (std::fflush(fout) != 0 || std::fclose(fout) != 0) die("write failed: %s", out_path.c_str());
    return total;
}

}  // namespace

int main(int argc, char* argv[]) {
    const Args args = parse_args(argc, argv);
    const size_t D = static_cast<size_t>(args.d);
    const size_t A = static_cast<size_t>(args.A);
    const size_t P = D / 2;

    std::printf("[1] loading dataset from %s ...\n", args.ds.c_str());
    const std::string base_path = args.ds + "/" + args.base_name;
    const std::string query_path = args.ds + "/" + args.query_name;
    const std::string base_attr_path = args.ds + "/" + args.base_attr_name;
    const std::string query_attr_path = args.ds + "/" + args.query_attr_name;

    const size_t nb = fvecs_probe(base_path, args.d);
    const size_t nq = fvecs_probe(query_path, args.d);
    const std::vector<uint64_t> salt = read_salt(args.salt_file, A, P);

    // base attrs read as uint32 (zero-extended), query attrs as int32
    // (sign-extended), matching Python.
    const std::vector<int64_t> attrs_b =
        attr_bin_read(base_attr_path, /*signed_payload=*/false, static_cast<int>(A), nb);
    const std::vector<int64_t> attrs_q =
        attr_bin_read(query_attr_path, /*signed_payload=*/true, static_cast<int>(A), nq);

    std::printf("      base %zux%zu query %zux%zu base_attr %zux%zu query_attr %zux%zu\n", nb, D,
                nq, D, nb, A, nq, A);
    std::printf("[2] building APR phase tables (A=%zu, P=%zu) ...\n", A, P);

    const std::map<TupleKey, TableEntry> tab_b = build_table(attrs_b, nb, A, P, salt);
    const std::map<TupleKey, TableEntry> tab_q = build_table(attrs_q, nq, A, P, salt);
    std::printf("      distinct base tuples=%zu query tuples=%zu\n", tab_b.size(), tab_q.size());

    if (mkdir(args.out.c_str(), 0777) != 0 && errno != EEXIST) {
        die("cannot create output directory: %s", args.out.c_str());
    }

    size_t chunk_rows = args.chunk_rows;
    if (chunk_rows == 0) chunk_rows = (64u << 20) / (4 + D * 4);  // 64MB/chunk
    if (chunk_rows == 0) chunk_rows = 1;

    std::printf("[3] apr-rotating base (A=%zu) ...\n", A);
    const NormStat sb = encode_apr_file(base_path, args.out + "/" + args.apr_base_name, args.d,
                                        A, attrs_b, nb, tab_b, args.threads, chunk_rows);
    std::printf("[4] apr-rotating query (A=%zu) ...\n", A);
    const NormStat sq = encode_apr_file(query_path, args.out + "/" + args.apr_query_name, args.d,
                                        A, attrs_q, nq, tab_q, args.threads, chunk_rows);

    const double mb = apr_io::norm_mean(sb);
    const double mq = apr_io::norm_mean(sq);
    std::printf("      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e\n", mb,
                apr_io::norm_std(sb), mq, apr_io::norm_std(sq));
    if (std::abs(mb - 1) >= 1e-3 || std::abs(mq - 1) >= 1e-3) {
        std::printf("[warn] encoded output is not unit-norm (inputs unnormalized); orthogonal rotation preserves distance, original-space GT stays valid.\n");
    }

    std::printf("[5] writing APR dataset + copying groundtruth ...\n");
    const std::string gt_src = args.ds + "/" + args.groundtruth_name;
    const std::string gt_dst = args.out + "/" + args.groundtruth_name;
    struct stat gst {};
    if (!args.groundtruth_name.empty() && stat(gt_src.c_str(), &gst) == 0) {
        copy_file(gt_src, gt_dst);
    } else if (!args.groundtruth_name.empty()) {
        std::printf("      [warn] groundtruth not found, skip copy: %s\n", gt_src.c_str());
    }

    std::printf("[ok] APR encoded -> %s base %zux%zu query %zux%zu\n", args.out.c_str(), nb, D,
                nq, D);
    return 0;
}
