#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/attribute/attr_rerank.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::symqg::QuantizedGraph<float>;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;
using rabitqlib::attribute::AttributeKind;
using rabitqlib::attribute::AttributeMatrix;
using rabitqlib::attribute::predicate_matches;
using rabitqlib::attribute::query_stride;
using rabitqlib::attribute::read_attr_bin;

struct Args {
    const char* index_file = nullptr;
    const char* query_file = nullptr;
    const char* gt_file = nullptr;
    std::string base_attr_file;
    std::string query_attr_file;
    size_t attr_dim = 0;
    std::string layout_text;
    std::vector<AttributeKind> layout;
    size_t qstride = 0;
    bool filtered = false;
    bool rerank = false;
    std::string orig_base_file;
    std::string orig_query_file;
    size_t rerank_factor = 10;
};

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <index> <query.fvecs> <gt.ivecs> [options]\n"
              << "  --base-attr F       uint32 [N,A] base attribute file\n"
              << "  --query-attr F      int32 [Nq,A] target attribute file\n"
              << "  --attr-dim A        number of attribute dimensions\n"
              << "  --layout T          comma-separated eq|range roles, one per A\n"
              << "  --rerank            enable exact rerank with oversampling\n"
              << "  --orig-base F       unrotated base fvecs (requires --rerank)\n"
              << "  --orig-query F      unrotated query fvecs (requires --rerank)\n"
              << "  --rerank-factor N   oversampling factor, default 10\n"
              << "The three attribute options enable equality filtering together.\n"
              << "With --layout range columns widen a query row to [lo,hi].\n";
}

size_t parse_size(const char* value, const std::string& option) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        throw std::invalid_argument("Invalid value for " + option);
    }
    return static_cast<size_t>(parsed);
}

// An empty layout keeps the original all-equality path. Otherwise every token
// declares one attribute column; range columns occupy [lo, hi] in the query
// file, so the row width is derived by the shared query_stride helper.
std::vector<AttributeKind> parse_layout(
    const std::string& text, size_t attr_dim
) {
    if (text.empty()) {
        return {};
    }
    std::vector<AttributeKind> layout;
    size_t start = 0;
    for (;;) {
        const size_t comma = text.find(',', start);
        const std::string token = text.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start
        );
        if (token == "eq") {
            layout.push_back(AttributeKind::kEquality);
        } else if (token == "range") {
            layout.push_back(AttributeKind::kRange);
        } else {
            throw std::invalid_argument("Unknown --layout token: " + token);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (layout.size() != attr_dim) {
        throw std::invalid_argument(
            "--layout token count must equal --attr-dim"
        );
    }
    return layout;
}

Args parse_args(int argc, char** argv) {
    Args args;
    args.index_file = argv[1];
    args.query_file = argv[2];
    args.gt_file = argv[3];
    bool factor_given = false;
    for (int i = 4; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--rerank") {
            args.rerank = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + option);
        }
        const char* value = argv[++i];
        if (option == "--base-attr") {
            args.base_attr_file = value;
        } else if (option == "--query-attr") {
            args.query_attr_file = value;
        } else if (option == "--attr-dim") {
            args.attr_dim = parse_size(value, option);
        } else if (option == "--layout") {
            args.layout_text = value;
        } else if (option == "--orig-base") {
            args.orig_base_file = value;
        } else if (option == "--orig-query") {
            args.orig_query_file = value;
        } else if (option == "--rerank-factor") {
            args.rerank_factor = parse_size(value, option);
            factor_given = true;
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }
    const bool any_attr = !args.base_attr_file.empty() ||
                          !args.query_attr_file.empty() || args.attr_dim != 0;
    if (any_attr && (args.base_attr_file.empty() || args.query_attr_file.empty() ||
                     args.attr_dim == 0)) {
        throw std::invalid_argument(
            "--base-attr, --query-attr and --attr-dim must be given together"
        );
    }
    args.filtered = any_attr;
    if (!args.layout_text.empty() && !args.filtered) {
        throw std::invalid_argument(
            "--layout requires --base-attr, --query-attr and --attr-dim"
        );
    }
    if (args.filtered) {
        args.layout = parse_layout(args.layout_text, args.attr_dim);
        args.qstride = query_stride(
            args.layout.empty() ? nullptr : args.layout.data(), args.attr_dim
        );
    }

    if (args.rerank &&
        (args.orig_base_file.empty() || args.orig_query_file.empty())) {
        throw std::invalid_argument(
            "--rerank requires both --orig-base and --orig-query"
        );
    }
    const bool any_orig = !args.orig_base_file.empty() ||
                          !args.orig_query_file.empty() || factor_given;
    if (!args.rerank && any_orig) {
        throw std::invalid_argument(
            "--orig-base, --orig-query and --rerank-factor require --rerank"
        );
    }
    return args;
}

std::vector<size_t> efs = {
    10,
    20,
    40,
    50,
    60,
    80,
    100,
    150,
    170,
    190,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    800,
    1500};
size_t test_round = 3;
size_t topk = 10;

int run(int argc, char** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const Args args = parse_args(argc, argv);

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(args.query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(args.gt_file, gt);
    const size_t nq = query.rows();
    const size_t total_count = nq * topk;

    index_type qg;
    qg.load(args.index_file);

    data_type orig_base;
    data_type orig_query;
    if (args.rerank) {
        if (qg.metric_type() != rabitqlib::METRIC_L2) {
            throw std::invalid_argument(
                "--rerank uses original-space L2 and requires an L2 index"
            );
        }
        rabitqlib::load_vecs<float, data_type>(
            args.orig_base_file.c_str(), orig_base
        );
        rabitqlib::load_vecs<float, data_type>(
            args.orig_query_file.c_str(), orig_query
        );
        if (orig_base.cols() != query.cols() ||
            orig_query.cols() != query.cols()) {
            throw std::runtime_error(
                "Original vector dimension does not match the query vectors"
            );
        }
        if (static_cast<size_t>(orig_query.rows()) != nq ||
            static_cast<size_t>(orig_base.rows()) < qg.num_points()) {
            throw std::runtime_error(
                "Original vector row count does not match the dataset"
            );
        }
    }

    size_t search_k = topk;
    if (args.rerank) {
        if (args.rerank_factor > std::numeric_limits<size_t>::max() / topk) {
            throw std::length_error("--rerank-factor is too large");
        }
        const size_t scaled = topk * args.rerank_factor;
        search_k = std::min(scaled, qg.num_points());
        if (search_k == 0 ||
            search_k > std::numeric_limits<uint32_t>::max()) {
            throw std::length_error("Invalid oversampled candidate count");
        }
    }

    AttributeMatrix<uint32_t> base_attrs;
    AttributeMatrix<int32_t> query_attrs;
    if (args.filtered) {
        base_attrs = read_attr_bin<uint32_t>(args.base_attr_file.c_str());
        query_attrs = read_attr_bin<int32_t>(args.query_attr_file.c_str());
        if (base_attrs.cols != args.attr_dim ||
            query_attrs.cols != args.qstride) {
            throw std::runtime_error(
                "Attribute column count does not match --attr-dim"
            );
        }
        if (base_attrs.rows < qg.num_points() || query_attrs.rows < nq) {
            throw std::runtime_error(
                "Attribute row count is smaller than the indexed dataset"
            );
        }
    }
    size_t predicate_violations = 0;

    rabitqlib::StopW stopw;

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));

    for (size_t r = 0; r < test_round; r++) {
        for (size_t i = 0; i < efs.size(); ++i) {
            const size_t ef = efs[i];
            size_t total_correct = 0;
            float total_time = 0;
            qg.set_ef(std::max(ef, search_k));
            std::vector<PID> results(total_count);
            std::vector<float> dists(total_count);
            std::vector<PID> candidates;
            std::vector<float> candidate_dists;
            if (args.rerank) {
                candidates.resize(nq * search_k);
                candidate_dists.resize(nq * search_k);
            }
            PID* search_ids = args.rerank ? candidates.data() : results.data();
            float* search_dists =
                args.rerank ? candidate_dists.data() : dists.data();
            const auto search_k_u32 = static_cast<uint32_t>(search_k);
            stopw.reset();
            if (args.filtered) {
                if (args.layout.empty()) {
                    qg.search_batch(
                        query.data(), nq, search_k_u32, search_ids, search_dists,
                        base_attrs.data(), query_attrs.data(), args.attr_dim
                    );
                } else {
                    qg.search_batch(
                        query.data(), nq, search_k_u32, search_ids, search_dists,
                        base_attrs.data(), query_attrs.data(), args.attr_dim,
                        args.layout.data()
                    );
                }
            } else {
                qg.search_batch(
                    query.data(), nq, search_k_u32, search_ids, search_dists
                );
            }
            if (args.rerank) {
                for (size_t z = 0; z < nq; ++z) {
                    rabitqlib::attribute::rerank_by_original_l2(
                        candidates.data() + z * search_k,
                        search_k,
                        orig_base.data(),
                        static_cast<size_t>(orig_base.rows()),
                        static_cast<size_t>(query.cols()),
                        orig_query.data() + z * static_cast<size_t>(query.cols()),
                        results.data() + z * topk,
                        dists.data() + z * topk,
                        topk,
                        rabitqlib::kPidMax
                    );
                }
            }
            total_time = stopw.get_elapsed_micro();
            for (size_t z = 0; z < nq; z++) {
                for (size_t y = 0; y < topk; y++) {
                    const PID id = results[z * topk + y];
                    if (id == rabitqlib::kPidMax) {
                        continue;
                    }
                    if (args.filtered) {
                        const size_t offset = static_cast<size_t>(id) * args.attr_dim;
                        if (!predicate_matches(
                                base_attrs.data() + offset,
                                query_attrs.data() + z * args.qstride,
                                args.attr_dim,
                                args.layout.empty() ? nullptr
                                                    : args.layout.data())) {
                            ++predicate_violations;
                        }
                    }
                    for (size_t k = 0; k < topk; k++) {
                        if (gt(static_cast<Eigen::Index>(z),
                               static_cast<Eigen::Index>(k)) == id) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }
            float qps = static_cast<float>(nq) / (total_time / 1e6F);
            float recall =
                static_cast<float>(total_correct) / static_cast<float>(total_count);

            all_qps[r][i] = qps;
            all_recall[r][i] = recall;
        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "EF\tQPS\tRecall\n";
    for (size_t i = 0; i < avg_qps.size(); ++i) {
        std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
    }
    if (args.filtered) {
        std::cout << "Predicate violations: " << predicate_violations << '\n';
    }
    if (args.rerank) {
        std::cout << "Rerank: factor=" << args.rerank_factor
                  << ", candidates=" << search_k << '\n';
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
