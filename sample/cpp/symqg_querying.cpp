#include <exception>
#include <iostream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <fstream>
#include <unordered_set>
#include <utility>

#include <getopt.h>
#include <omp.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/utils/attr_filter.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::symqg::QuantizedGraph<float>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

// 内置默认值：保持与 example.sh 原有用法（topk=10，内置 ef 列表）兼容。
// run.sh 会通过位置参数覆盖这些默认值。
std::vector<size_t> DEFAULT_EFS = {
    10, 20, 40, 50, 60, 80, 100, 150, 170, 190,
    200, 250, 300, 400, 500, 600, 700, 800, 1500};
size_t DEFAULT_TOPK = 10;
size_t DEFAULT_ROUNDS = 3;
std::string DEFAULT_ALGO = "symqg";
size_t DEFAULT_THREADS = 1;

// 解析逗号分隔的 ef 列表（如 "10,20,40"），跳过空项与非正值。
std::vector<size_t> parse_efs(const std::string& s) {
    std::vector<size_t> out;
    if (s.empty()) return out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(0, tok.find_first_not_of(" \t"));
        tok.erase(tok.find_last_not_of(" \t") + 1);
        if (tok.empty()) continue;
        size_t v = static_cast<size_t>(std::stoul(tok));
        if (v > 0) out.push_back(v);
    }
    return out;
}

// 原始空间 L2² 距离（重排用，与图内旋转空间距离解耦）。
static inline float l2sq(const float* a, const float* b, size_t d) {
    float s = 0.f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

// ===================== 属性加载相关（行间过滤）=====================
// 自适应加载属性 bin：文件支持两种格式
//   1) 纯数据（无头）：filesize == expected_n * A * sizeof(T)
//   2) 带头自描述：前 8 字节为两个 int32 [N, A]，filesize == 8 + 数据区
// 已知 expected_n 与 A 时按文件大小自动判断；若有头则额外校验 N/A。
// 谓词函数 attr_match / attr_match_range 由 rabitqlib/utils/attr_filter.hpp 提供。
template <typename T>
void load_attr_bin(const char* path, size_t expected_n, uint32_t A, std::vector<T>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[error] cannot open " << path << std::endl;
        exit(-1);
    }
    size_t data_bytes = (size_t)expected_n * A * sizeof(T);
    in.seekg(0, std::ios::end);
    std::ios::pos_type total = in.tellg();
    in.seekg(0, std::ios::beg);

    bool has_header = false;
    if ((size_t)total == data_bytes) {
        has_header = false;  // 纯数据格式
    } else if ((size_t)total == sizeof(int32_t) * 2 + data_bytes) {
        has_header = true;   // 8 字节 [N,A] 头
        int32_t hdr[2];
        in.read((char*)hdr, sizeof(int32_t) * 2);
        if (hdr[0] != (int32_t)expected_n) {
            std::cerr << "[error] attr N mismatch in " << path
                      << ": header N=" << hdr[0] << " but expected " << expected_n << std::endl;
            exit(-1);
        }
        if ((uint32_t)hdr[1] != A) {
            std::cerr << "[error] attr A mismatch in " << path
                      << ": header A=" << hdr[1] << " but expected " << A << std::endl;
            exit(-1);
        }
    } else {
        std::cerr << "[error] size mismatch " << path << ": expect "
                  << data_bytes << " or "
                  << (sizeof(int32_t) * 2 + data_bytes)
                  << " bytes, got " << (size_t)total << std::endl;
        exit(-1);
    }
    (void)has_header;  // 纯数据/带头均按相同方式读取数据区
    out.resize((size_t)expected_n * A);
    in.read((char*)out.data(), (std::streamsize)data_bytes);
    in.close();
}

int run(int argc, char** argv) {
    // ---- 参数解析：位置参数 index / query / gt 在前，其余为可选 flag ----
    char* index_file = nullptr;
    char* query_file = nullptr;
    char* gt_file = nullptr;
    size_t topk = DEFAULT_TOPK;
    std::string efs_csv;
    std::string algo = DEFAULT_ALGO;
    size_t rounds = DEFAULT_ROUNDS;
    size_t threads = DEFAULT_THREADS;

    // 行间过滤（inline filter）相关参数
    char base_attr_path[1024] = "";
    char query_attr_path[1024] = "";
    uint32_t attr_dim = 0;      // 属性维数 A（由 -A 传入；行间过滤必填）
    bool range_mode = false;    // 范围模式开关（-r 启用）

    // 增强臂（参考 RaBitQ-range-inline-rot 的新过滤算法）参数
    //   --range_m M            : 每条逻辑查询的子中心旋转查询数（M>=1；=1 退化为单次）
    //   --orig_base / --orig_query : 非旋转原始向量，供重排 L2² 计算（图内存储的是旋转向量）
    char orig_base_path[2048] = "";
    char orig_query_path[2048] = "";
    uint32_t range_m = 0;

    static struct option longopts[] = {
        {"base_attr",  required_argument, 0, 'b'},
        {"query_attr", required_argument, 0, 'a'},
        {"attr_dim",   required_argument, 0, 'A'},
        {"range",      no_argument,       0, 'r'},
        {"range_m",    required_argument, 0, 'M'},
        {"orig_base",  required_argument, 0, 1001},
        {"orig_query", required_argument, 0, 1002},
        {0, 0, 0, 0}
    };
    int iarg = 0, ind = 0;
    opterr = 1;
    while ((iarg = getopt_long(argc, argv, "b:a:A:rM:", longopts, &ind)) != -1) {
        switch (iarg) {
            case 'b':
                if (optarg) std::strncpy(base_attr_path, optarg, sizeof(base_attr_path) - 1);
                break;
            case 'a':
                if (optarg) std::strncpy(query_attr_path, optarg, sizeof(query_attr_path) - 1);
                break;
            case 'A':
                if (optarg) attr_dim = (uint32_t)std::atoi(optarg);
                break;
            case 'r':
                range_mode = true;
                break;
            case 'M':
                if (optarg) range_m = (uint32_t)std::atoi(optarg);
                break;
            case 1001:
                if (optarg) std::strncpy(orig_base_path, optarg, sizeof(orig_base_path) - 1);
                break;
            case 1002:
                if (optarg) std::strncpy(orig_query_path, optarg, sizeof(orig_query_path) - 1);
                break;
            default:
                break;
        }
    }
    // 剩余的位置参数：index query gt [topk] [efs_csv] [algo] [rounds] [threads]
    int pos = optind;
    if (argc - pos < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <index> <query.fvecs> <gt.ivecs>"
                  << " [topk] [efs_csv] [algo] [rounds] [threads]"
                  << " [-b base_attr.bin] [-a query_attr.bin] [-A attr_dim] [-r]"
                  << " [--range_m M] [--orig_base base.fvecs] [--orig_query query.fvecs]\n";
        return 1;
    }
    index_file = argv[pos];
    query_file = argv[pos + 1];
    gt_file = argv[pos + 2];
    if (argc - pos > 3 && argv[pos + 3][0] != '\0')
        topk = static_cast<size_t>(std::stoul(argv[pos + 3]));
    if (argc - pos > 4 && argv[pos + 4][0] != '\0')
        efs_csv = argv[pos + 4];
    if (argc - pos > 5 && argv[pos + 5][0] != '\0')
        algo = argv[pos + 5];
    if (argc - pos > 6 && argv[pos + 6][0] != '\0')
        rounds = static_cast<size_t>(std::stoul(argv[pos + 6]));
    if (argc - pos > 7 && argv[pos + 7][0] != '\0')
        threads = static_cast<size_t>(std::stoul(argv[pos + 7]));

    std::vector<size_t> efs = efs_csv.empty() ? DEFAULT_EFS : parse_efs(efs_csv);
    if (efs.empty()) efs = DEFAULT_EFS;

    // 增强臂判定：range_m>0 且提供原始 base/query 即启用「旋转+M 子中心+重排闸门」
    const bool enhanced =
        (range_m > 0) &&
        (orig_base_path[0] != '\0') && (orig_query_path[0] != '\0');
    if (enhanced) {
        std::cerr << "[enhanced] 启用范围旋转增强算法: range_m=" << range_m << std::endl;
        range_mode = true;   // 增强臂的 -a 是范围 bin（2*A 列），统一按 range 解析
    }

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    const size_t nq_rows = query.rows();
    // 增强臂下主查询文件是 Nq*M 行旋转 query；逻辑查询数 = 行数 / range_m。
    const size_t nq_eff = enhanced ? (nq_rows / range_m) : nq_rows;
    const size_t total_count = nq_eff * topk;

    index_type qg;
    qg.load(index_file);

    // ---- 加载属性（行间过滤 / 增强臂开启时）----
    const bool use_attr =
        (base_attr_path[0] != '\0') || (query_attr_path[0] != '\0');
    std::vector<uint32_t> base_attr;   // [N][A]
    std::vector<int32_t> query_attr;   // [Nq_logical][A]（等值）或 [Nq_logical][2A]（范围）
    uint32_t A = 0;
    if (use_attr) {
        if (base_attr_path[0] == '\0' || query_attr_path[0] == '\0') {
            std::cerr << "[error] base_attr (-b) and query_attr (-a) must be provided together\n";
            return -1;
        }
        if (attr_dim == 0) {
            std::cerr << "[error] attr_dim (-A) must be provided for inline filter\n";
            return -1;
        }
        A = attr_dim;
        uint32_t query_cols = range_mode ? 2 * A : A;
        // 增强臂下主查询文件是 Nq*M 行，但 range bin 仅有 Nq_logical 行 -> 用 nq_eff 校验
        load_attr_bin(base_attr_path, qg.num_vertices(), A, base_attr);
        load_attr_bin(query_attr_path, nq_eff, query_cols, query_attr);
        std::cerr << "Loaded attributes: A = " << A
                  << ", base rows = " << qg.num_vertices() << ", logical query rows = " << nq_eff
                  << (range_mode ? " [range mode: query cols = 2*A]" : "") << std::endl;
    }

    // ---- 原始（非旋转）base/query 供重排 L2²（增强臂与 inline 精确重排复用；
    //      旋转保距，故旋转空间距离==原始空间距离，与 GT 口径一致）----
    data_type orig_base, orig_query;
    size_t D = query.cols();
    const bool have_orig =
        (orig_base_path[0] != '\0') && (orig_query_path[0] != '\0');
    if (have_orig) {
        rabitqlib::load_vecs<float, data_type>(orig_base_path, orig_base);
        rabitqlib::load_vecs<float, data_type>(orig_query_path, orig_query);
        D = orig_base.cols();
        std::cerr << "Loaded orig vectors for rerank: base=" << orig_base.rows()
                  << " query=" << orig_query.rows() << " dim=" << D << std::endl;
        if (orig_base.rows() != qg.num_vertices()) {
            std::cerr << "[error] orig_base rows (" << orig_base.rows()
                      << ") != graph vertices (" << qg.num_vertices() << ")\n";
            return -1;
        }
        if (orig_query.rows() != nq_eff) {
            std::cerr << "[error] orig_query rows (" << orig_query.rows()
                      << ") != logical queries (" << nq_eff << ")\n";
            return -1;
        }
    }

    rabitqlib::StopW stopw;

    std::vector<std::vector<float>> all_qps(rounds,
                                            std::vector<float>(efs.size()));
    std::vector<std::vector<float>> all_recall(rounds,
                                               std::vector<float>(efs.size()));
    // 行间过滤模式下额外统计：谓词违反数（校验内核正确性）与完全满足查询数
    std::vector<std::vector<uint32_t>> all_viol(rounds,
                                                std::vector<uint32_t>(efs.size()));
    std::vector<std::vector<uint32_t>> all_sat(rounds,
                                               std::vector<uint32_t>(efs.size()));

    for (size_t r = 0; r < rounds; r++) {
        for (size_t i = 0; i < efs.size(); ++i) {
            size_t ef = efs[i];
            // 行间过滤 / 增强臂：ef 作为图的探索预算，无需过采样；knn 直接取 K。
            // Symqg 要求 search_ef >= knn，故 search_ef = max(ef, K)。
            const size_t search_ef = ef < topk ? topk : ef;
            // 增强臂：每个子中心需收集多于 K 的候选（旋转空间近邻未必等于原始 L2 近邻，
            // 须扩大候选池再按原始空间 L2² 重排，否则 top-K 召回受限）。图的 ef 须 >= knn。
            const size_t cand_knn = enhanced
                ? std::min((size_t)qg.num_vertices(),
                           std::max((size_t)topk * 10, search_ef))
                : (size_t)topk;
            const size_t ef_for_search = enhanced
                ? std::max(search_ef, cand_knn)
                : search_ef;

            size_t total_correct = 0;
            uint32_t total_viol = 0;
            uint32_t n_sat = 0;
            qg.set_ef(ef_for_search);
            stopw.reset();
            const int nt =
                threads == 0 ? omp_get_max_threads()
                            : static_cast<int>(threads);
#pragma omp parallel for num_threads(nt) schedule(dynamic) if (nt > 1) reduction(+:total_correct, total_viol, n_sat)
            for (std::ptrdiff_t z = 0;
                 z < static_cast<std::ptrdiff_t>(nq_eff); ++z) {
                // results/dists 必须线程私有，避免多线程写同一缓冲造成数据竞争。
                std::vector<PID> results(topk);
                std::vector<float> dists(topk);
                const int32_t* qa = nullptr;

                if (enhanced) {
                    // ---- 增强臂：M 条旋转子查询并集候选（候选池放大以覆盖原始 L2 近邻）----
                    std::unordered_set<uint32_t> cand;
                    std::vector<PID> tmp_r(cand_knn);
                    std::vector<float> tmp_d(cand_knn);
                    for (uint32_t s = 0; s < range_m; ++s) {
                        qg.search(&query(static_cast<Eigen::Index>(
                                         z * (std::ptrdiff_t)range_m + s), 0),
                                  (uint32_t)cand_knn, tmp_r.data(), tmp_d.data());
                        for (size_t y = 0; y < cand_knn; ++y)
                            if (tmp_r[y] != std::numeric_limits<uint32_t>::max())
                                cand.insert(tmp_r[y]);
                    }
                    // ---- 原始空间 L2² 重排 + 精确 range 闸门（Soundness 兜底）----
                    const int32_t* qr = &query_attr[(std::ptrdiff_t)z * (std::ptrdiff_t)(2 * A)];
                    std::vector<std::pair<float, uint32_t>> re;
                    re.reserve(cand.size());
                    for (uint32_t id : cand) {
                        if (!rabitqlib::utils::attr_match_range(
                                &base_attr[(std::ptrdiff_t)id * A], qr, A))
                            continue;   // 硬闸门：不通过丢弃（Soundness 保证）
                        float d2 = l2sq(&orig_base(static_cast<Eigen::Index>(id), 0),
                                        &orig_query(static_cast<Eigen::Index>(z), 0), D);
                        re.emplace_back(d2, id);
                    }
                    std::sort(re.begin(), re.end());
                    size_t take = std::min((size_t)topk, re.size());
                    for (size_t y = 0; y < take; ++y) {
                        results[y] = re[y].second;
                        dists[y] = re[y].first;
                    }
                    for (size_t y = take; y < topk; ++y) {
                        results[y] = std::numeric_limits<uint32_t>::max();
                        dists[y] = std::numeric_limits<float>::max();
                    }
                } else if (use_attr) {
                    // ---- 现有行间过滤（等值 / -r 范围硬闸门）----
                    qa = query_attr.data() +
                         (std::ptrdiff_t)z * (range_mode ? (std::ptrdiff_t)(2 * A)
                                                        : (std::ptrdiff_t)A);
                    // 属性谓词在内核中即时判定，结果堆只含匹配项。
                    qg.search(&query(static_cast<Eigen::Index>(z), 0),
                              (uint32_t)topk, results.data(), dists.data(),
                              base_attr.data(), qa, A, range_mode);
                    // 精确重排（对齐 IVF 阶段2）：仅对内核已筛选出的匹配候选，
                    // 按原始空间 L2² 重排，修正 bit>0 量化距离噪声导致的错误排序。
                    // 非匹配节点不进入候选，故不会计算其原始距离（满足 2.1）。
                    if (have_orig) {
                        std::vector<std::pair<float, uint32_t>> re;
                        re.reserve(topk);
                        for (size_t y = 0; y < topk; ++y) {
                            PID pid = results[y];
                            if (pid == std::numeric_limits<uint32_t>::max()) continue;
                            float d2 = l2sq(&orig_base(static_cast<Eigen::Index>(pid), 0),
                                            &orig_query(static_cast<Eigen::Index>(z), 0), D);
                            re.emplace_back(d2, pid);
                        }
                        std::sort(re.begin(), re.end());
                        for (size_t y = 0; y < topk; ++y) {
                            if (y < re.size()) {
                                results[y] = re[y].second;
                                dists[y] = re[y].first;
                            } else {
                                results[y] = std::numeric_limits<uint32_t>::max();
                                dists[y] = std::numeric_limits<float>::max();
                            }
                        }
                    }
                } else {
                    // ---- 纯向量检索 ----
                    qg.search(&query(static_cast<Eigen::Index>(z), 0),
                              (uint32_t)topk, results.data(), dists.data());
                }

                size_t correct = 0;
                uint32_t viol = 0;
                for (size_t y = 0; y < topk; y++) {
                    PID pid = results[y];
                    if (pid == std::numeric_limits<uint32_t>::max()) continue;  // 哨兵（匹配不足 K）
                    // 增强臂结果已通过精确闸门，无需重复校验（viol 恒为 0）
                    if (!enhanced && use_attr && qa != nullptr) {
                        const uint32_t* ba = &base_attr[(std::ptrdiff_t)pid * A];
                        bool ok = range_mode
                            ? rabitqlib::utils::attr_match_range(ba, qa, A)
                            : rabitqlib::utils::attr_match(ba, qa, A);
                        if (!ok) viol++;
                    }
                    for (size_t k = 0; k < topk; k++) {
                        if (gt(static_cast<Eigen::Index>(z),
                               static_cast<Eigen::Index>(k)) == pid) {
                            correct++;
                            break;
                        }
                    }
                }
                total_correct += correct;
                total_viol += viol;
                if (viol == 0) n_sat++;
            }
            double elapsed_sec = stopw.get_elapsed_micro() / 1e6;
            float qps = elapsed_sec > 0
                            ? static_cast<float>(nq_eff) /
                                  static_cast<float>(elapsed_sec)
                            : 0.0F;
            float recall = static_cast<float>(total_correct) /
                           static_cast<float>(total_count);

            all_qps[r][i] = qps;
            all_recall[r][i] = recall;
            all_viol[r][i] = total_viol;
            all_sat[r][i] = n_sat;
        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);
    auto avg_sat = rabitqlib::horizontal_avg(all_sat);
    auto avg_viol = rabitqlib::horizontal_avg(all_viol);

    std::cout << "EF\tQPS\tRecall\n";
    for (size_t i = 0; i < avg_qps.size(); ++i) {
        std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i]
                  << '\n';
    }

    // ---- eval_tradeoff.py 兼容行 ----
    // 格式：[RESULT] algo=<alg> param=<ef> k=<K> qps=<q> recall=<r>
    for (size_t i = 0; i < avg_qps.size(); ++i) {
        std::cout << "[RESULT] algo=" << algo
                  << " param=" << efs[i]
                  << " k=" << topk
                  << " qps=" << avg_qps[i]
                  << " recall=" << avg_recall[i] << '\n';
        if (use_attr || enhanced) {
            std::cout << "Filtered results all satisfy attr: "
                      << (uint32_t)avg_sat[i] << " / " << nq_eff << '\n';
            std::cout << "Predicate violations: " << (uint32_t)avg_viol[i] << '\n';
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
