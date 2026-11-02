// ============================================================================
// encode_fdm.cpp — 属性 FDM/CDMA 相位叠加编码器（C++ 版，替代 encode_attr_fdm.py）
//
// 功能：对统一数据集施加 FDM 旋转编码（每维对同时承载全部 A 个属性的相位码，
//       正交旋转、保距，原始空间 GT 与精确重排仍有效）。base/query 均施加同一
//       套由 salt 决定的相位，故二者一致、GT 仍有效。
//
// 与 Python encode_attr_fdm.py / _dataset_utils.encode_fdm 的关系：
//   - CLI 完全兼容（另加 --salt-file / --threads）；run.sh 调用方式不变。
//   - 相位哈希位级复刻 _hash_to_phases_f0（splitmix64 终化器 _mix64）：
//         base = mix64(v * GOLDEN)
//         h    = mix64(base ^ salt[j]) & 0x7FFFFFFF
//         phase = (h / 2147483647.0) * 2π
//     salt 由 --salt-file 提供（numpy 精确生成的 A×P 个 int64，保证与现有
//     sift/msmarco/openai1536 的 FDM 结果连续、重跑字节一致）。
//   - 数值路径：float32 读入 -> double 运算 -> cast float32；不归一化（与
//     Python 一致，仅打印非单位范数告警）。
//
// 性能：用「属性元组 (v_0..v_{A-1}) -> 总相位 cos/sin 表」预计算（本配置
//       C^A ≤ 1000），每行 O(1) 查表后对全部维度对施加旋转；流式分块 +
//       多线程按行并行（64MB/chunk），峰值内存 O(chunk + 表)，与文件大小无关。
//
// 编译（与参考实现一致）：
//   g++ -std=c++17 -o ../bin/encode_fdm encode_fdm.cpp -I . \
//       -Ofast -march=core-avx2 -pthread -fno-fast-math -ffp-contract=off
// 注意：-fno-fast-math 必须（pairwise 范数求和依赖严格浮点结合序，否则与
//       numpy 位级不一致）。
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

#include "rot_encode_io.hpp"

namespace {

using rot_io::die;
using rot_io::fvecs_probe;
using rot_io::attr_bin_read;
using rot_io::copy_file;
using rot_io::NormStat;

const uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
const double k2Pi = 2.0 * 3.14159265358979323846;

// 位级复刻 _dataset_utils._mix64（splitmix64 终化器）。
inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// 位级复刻 _hash_to_phases_f0 的单相位：v 已为非负属性值，saltj 为 uint64 位模式。
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
    std::string fdm_base_name = "fdm_base.fvecs";
    std::string fdm_query_name = "fdm_query.fvecs";
    std::string salt_file;
    size_t threads = 0, chunk_rows = 0;
};

void usage(const char* prog) {
    std::fprintf(stderr,
                 "用法: %s --ds DIR --out DIR --d D --A A --salt-file F [选项]\n"
                 "  --ds DIR              输入统一数据集目录\n"
                 "  --out DIR             输出 FDM 目录\n"
                 "  --d D                 向量维度\n"
                 "  --A A                 属性数\n"
                 "  --enc-seed S          编码种子（仅作记录；salt 由 --salt-file 提供）\n"
                 "  --base-name F         输入 base 向量文件名\n"
                 "  --query-name F        输入 query 向量文件名\n"
                 "  --base-attr-name F    输入 base 属性文件名（uint32 自描述 bin）\n"
                 "  --query-attr-name F   输入 query 属性文件名（int32 自描述 bin）\n"
                 "  --groundtruth-name F  输入 groundtruth 文件名（存在则拷贝到 out）\n"
                 "  --fdm-base-name F     输出 FDM base 文件名\n"
                 "  --fdm-query-name F    输出 FDM query 文件名\n"
                 "  --salt-file F         numpy 精确生成的 A×P 个 int64 salt（必须）\n"
                 "  --threads N           并行线程数（默认 hardware_concurrency）\n"
                 "  --chunk-rows N        每块行数（默认按 64MB/chunk 推导）\n"
                 "  -h, --help            显示帮助\n",
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
        {"queries-name", required_argument, nullptr, 11},  // 兼容 Python 版，忽略
        {"fdm-base-name", required_argument, nullptr, 12},
        {"fdm-query-name", required_argument, nullptr, 13},
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
            case 11: break;  // queries-name 忽略
            case 12: a.fdm_base_name = optarg; break;
            case 13: a.fdm_query_name = optarg; break;
            case 14: a.salt_file = optarg; break;
            case 15: a.threads = static_cast<size_t>(std::atoll(optarg)); break;
            case 16: a.chunk_rows = static_cast<size_t>(std::atoll(optarg)); break;
            case 'h': usage(argv[0]); std::exit(0);
            default: usage(argv[0]); std::exit(1);
        }
    }
    if (a.ds.empty()) die("缺少 --ds");
    if (a.out.empty()) die("缺少 --out");
    if (a.d < 2) die("--d 必须 >= 2（旋转作用于维度对），当前 %d", a.d);
    if (a.A < 1) die("--A 必须 >= 1，当前 %d", a.A);
    if (a.salt_file.empty()) die("缺少 --salt-file（须 numpy 精确生成）");
    if (a.threads == 0) a.threads = std::thread::hardware_concurrency();
    if (a.threads == 0) a.threads = 1;
    return a;
}

// 读 salt：A×P 个 int64（小端）， reinterpret 为 uint64 位模式（与 numpy int64 一致）。
std::vector<uint64_t> read_salt(const std::string& path, size_t A, size_t P) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) die("无法打开 salt 文件: %s", path.c_str());
    std::vector<int64_t> tmp(A * P);
    if (std::fread(tmp.data(), sizeof(int64_t), A * P, f) != A * P) {
        std::fclose(f);
        die("salt 文件长度不足（期望 %zu 个 int64）: %s", A * P, path.c_str());
    }
    std::fclose(f);
    std::vector<uint64_t> salt(A * P);
    for (size_t i = 0; i < A * P; ++i) salt[i] = static_cast<uint64_t>(tmp[i]);
    return salt;
}

using TupleKey = std::vector<uint32_t>;

struct TableEntry {
    std::vector<double> c;  // 长度 P
    std::vector<double> s;  // 长度 P
};

// 单次遍历属性矩阵，构建「元组 -> 总相位 cos/sin 表」。
// 元组 = 一行各属性值 (v_0..v_{A-1})；total[j] = Σ_a phase_f0(v_a, salt[a*P+j])。
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

// 流式分块 + 多线程按行并行施加 FDM 旋转（表只读，无锁）；返回范数统计。
NormStat encode_fdm_file(const std::string& in_path, const std::string& out_path, int d,
                         size_t A, const std::vector<int64_t>& attrs, size_t n,
                         const std::map<TupleKey, TableEntry>& tab, size_t threads,
                         size_t chunk_rows) {
    const size_t D = static_cast<size_t>(d);
    const size_t P = D / 2;
    const size_t row_bytes = 4 + D * 4;
    FILE* fin = std::fopen(in_path.c_str(), "rb");
    if (fin == nullptr) die("无法打开输入: %s", in_path.c_str());
    FILE* fout = std::fopen(out_path.c_str(), "wb");
    if (fout == nullptr) die("无法打开输出: %s", out_path.c_str());
    std::vector<char> buf(chunk_rows * row_bytes);
    std::vector<char> obuf(1 << 20);
    std::setvbuf(fout, obuf.data(), _IOFBF, obuf.size());

    NormStat total{};
    for (size_t r0 = 0; r0 < n; r0 += chunk_rows) {
        const size_t rows = (n - r0 < chunk_rows) ? (n - r0) : chunk_rows;
        if (std::fread(buf.data(), row_bytes, rows, fin) != rows) {
            die("读取失败: %s（行 %zu 处）", in_path.c_str(), r0);
        }
        for (size_t i = 0; i < rows; ++i) {
            int32_t h = 0;
            std::memcpy(&h, buf.data() + i * row_bytes, 4);
            if (h != d) die("fvecs 行 %zu 维度头=%d 期望 %d: %s", r0 + i, h, d, in_path.c_str());
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
                    if (it == tab.end()) die("未找到元组（表构建遗漏）: 行 %zu", r0 + i);
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
            die("写出失败: %s", out_path.c_str());
        }
        for (const auto& s : stats) {
            total.sum += s.sum;
            total.sum2 += s.sum2;
            total.count += s.count;
        }
    }
    std::fclose(fin);
    if (std::fflush(fout) != 0 || std::fclose(fout) != 0) die("写出失败: %s", out_path.c_str());
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

    // base 属性按 uint32 读（零扩展），query 属性按 int32 读（符号扩展），与 Python 一致。
    const std::vector<int64_t> attrs_b =
        attr_bin_read(base_attr_path, /*signed_payload=*/false, static_cast<int>(A), nb);
    const std::vector<int64_t> attrs_q =
        attr_bin_read(query_attr_path, /*signed_payload=*/true, static_cast<int>(A), nq);

    std::printf("      base %zux%zu query %zux%zu base_attr %zux%zu query_attr %zux%zu\n", nb, D,
                nq, D, nb, A, nq, A);
    std::printf("[2] building FDM phase tables (A=%zu, P=%zu) ...\n", A, P);

    const std::map<TupleKey, TableEntry> tab_b = build_table(attrs_b, nb, A, P, salt);
    const std::map<TupleKey, TableEntry> tab_q = build_table(attrs_q, nq, A, P, salt);
    std::printf("      distinct base tuples=%zu query tuples=%zu\n", tab_b.size(), tab_q.size());

    if (mkdir(args.out.c_str(), 0777) != 0 && errno != EEXIST) {
        die("无法创建输出目录: %s", args.out.c_str());
    }

    size_t chunk_rows = args.chunk_rows;
    if (chunk_rows == 0) chunk_rows = (64u << 20) / (4 + D * 4);  // 64MB/chunk
    if (chunk_rows == 0) chunk_rows = 1;

    std::printf("[3] fdm-rotating base (A=%zu) ...\n", A);
    const NormStat sb = encode_fdm_file(base_path, args.out + "/" + args.fdm_base_name, args.d,
                                        A, attrs_b, nb, tab_b, args.threads, chunk_rows);
    std::printf("[4] fdm-rotating query (A=%zu) ...\n", A);
    const NormStat sq = encode_fdm_file(query_path, args.out + "/" + args.fdm_query_name, args.d,
                                        A, attrs_q, nq, tab_q, args.threads, chunk_rows);

    const double mb = rot_io::norm_mean(sb);
    const double mq = rot_io::norm_mean(sq);
    std::printf("      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e\n", mb,
                rot_io::norm_std(sb), mq, rot_io::norm_std(sq));
    if (std::abs(mb - 1) >= 1e-3 || std::abs(mq - 1) >= 1e-3) {
        std::printf("[warn] 编码输出非单位范数（输入未归一化）；正交旋转保距，原始空间 GT 仍有效。\n");
    }

    std::printf("[5] writing FDM dataset + copying groundtruth ...\n");
    const std::string gt_src = args.ds + "/" + args.groundtruth_name;
    const std::string gt_dst = args.out + "/" + args.groundtruth_name;
    struct stat gst {};
    if (!args.groundtruth_name.empty() && stat(gt_src.c_str(), &gst) == 0) {
        copy_file(gt_src, gt_dst);
    } else if (!args.groundtruth_name.empty()) {
        std::printf("      [warn] groundtruth not found, skip copy: %s\n", gt_src.c_str());
    }

    std::printf("[ok] FDM encoded -> %s base %zux%zu query %zux%zu\n", args.out.c_str(), nb, D,
                nq, D);
    return 0;
}
