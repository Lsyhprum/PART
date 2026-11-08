#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/attribute/attr_rerank.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf::IVF;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;
using rabitqlib::attribute::AttributeKind;
using rabitqlib::attribute::AttributeMatrix;
using rabitqlib::attribute::predicate_matches;
using rabitqlib::attribute::query_stride;
using rabitqlib::attribute::read_attr_bin;

static size_t topk = 100;
static size_t test_round = 5;

struct Args {
    const char* index_file = nullptr;
    const char* query_file = nullptr;
    const char* gt_file = nullptr;
    std::optional<bool> use_hacc;
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
              << " <index> <query.fvecs> <gt.ivecs> [hacc] [options]\n"
              << "  hacc                  optional \"true\"|\"false\", else automatic\n"
              << "  --base-attr F         uint32 [N,A] base attribute file\n"
              << "  --query-attr F        int32 [Nq,A] target attribute file\n"
              << "  --attr-dim A          number of attribute dimensions\n"
              << "  --layout T            comma-separated eq|range roles, one per A\n"
              << "  --rerank              enable exact rerank with oversampling\n"
              << "  --orig-base F         unrotated base fvecs (requires --rerank)\n"
              << "  --orig-query F        unrotated query fvecs (requires --rerank)\n"
              << "  --rerank-factor N     oversampling factor, default 10\n"
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

    int i = 4;
    if (i < argc && std::string(argv[i]).rfind("--", 0) != 0) {
        const std::string hacc_str(argv[i]);
        args.use_hacc = hacc_str != "false";
        if (!*args.use_hacc) {
            std::cout << "Do not use Hacc FastScan\n";
        }
        ++i;
    }

    bool factor_given = false;
    for (; i < argc; ++i) {
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

struct Outcome {
    float recall;
    float elapsed_us;
    size_t violations;
};

// One search, optional rerank and scoring. The nprobe sweep and the timed
// measurement loop share this path so both see the same predicates, candidate
// count and rerank policy.
class SearchHarness {
   public:
    SearchHarness(
        const index_type& ivf,
        const data_type& query,
        const gt_type& gt,
        const Args& args,
        const AttributeMatrix<uint32_t>& base_attrs,
        const AttributeMatrix<int32_t>& query_attrs,
        const data_type& orig_base,
        const data_type& orig_query,
        size_t nq,
        size_t search_k
    )
        : ivf_(ivf),
          query_(query),
          gt_(gt),
          args_(args),
          base_attrs_(base_attrs),
          query_attrs_(query_attrs),
          orig_base_(orig_base),
          orig_query_(orig_query),
          nq_(nq),
          search_k_(search_k),
          results_(nq * topk),
          dists_(nq * topk) {
        if (args_.rerank) {
            candidates_.resize(nq * search_k);
            candidate_dists_.resize(nq * search_k);
        }
    }

    Outcome run(size_t nprobe) {
        PID* search_ids = args_.rerank ? candidates_.data() : results_.data();
        float* search_dists =
            args_.rerank ? candidate_dists_.data() : dists_.data();

        rabitqlib::StopW stopw;
        if (args_.filtered) {
            if (args_.layout.empty()) {
                ivf_.search_batch(
                    query_.data(), nq_, search_k_, nprobe, search_ids, search_dists,
                    base_attrs_.data(), query_attrs_.data(), args_.attr_dim,
                    args_.use_hacc
                );
            } else {
                ivf_.search_batch(
                    query_.data(), nq_, search_k_, nprobe, search_ids, search_dists,
                    base_attrs_.data(), query_attrs_.data(), args_.attr_dim,
                    args_.layout.data(), args_.use_hacc
                );
            }
        } else {
            ivf_.search_batch(
                query_.data(), nq_, search_k_, nprobe, search_ids, search_dists,
                args_.use_hacc
            );
        }
        if (args_.rerank) {
            const size_t dim = static_cast<size_t>(query_.cols());
            for (size_t z = 0; z < nq_; ++z) {
                rabitqlib::attribute::rerank_by_original_l2(
                    candidates_.data() + z * search_k_,
                    search_k_,
                    orig_base_.data(),
                    static_cast<size_t>(orig_base_.rows()),
                    dim,
                    orig_query_.data() + z * dim,
                    results_.data() + z * topk,
                    dists_.data() + z * topk,
                    topk,
                    rabitqlib::kPidMax
                );
            }
        }
        const float elapsed_us = stopw.get_elapsed_micro();

        size_t correct = 0;
        size_t violations = 0;
        const size_t gt_cols =
            std::min<size_t>(topk, static_cast<size_t>(gt_.cols()));
        for (size_t z = 0; z < nq_; ++z) {
            for (size_t y = 0; y < topk; ++y) {
                const PID id = results_[z * topk + y];
                if (id == rabitqlib::kPidMax) {
                    continue;
                }
                if (args_.filtered &&
                    !predicate_matches(
                        base_attrs_.data() +
                            static_cast<size_t>(id) * args_.attr_dim,
                        query_attrs_.data() + z * args_.qstride,
                        args_.attr_dim,
                        args_.layout.empty() ? nullptr
                                             : args_.layout.data())) {
                    ++violations;
                }
                for (size_t k = 0; k < gt_cols; ++k) {
                    if (gt_(static_cast<Eigen::Index>(z),
                            static_cast<Eigen::Index>(k)) == id) {
                        ++correct;
                        break;
                    }
                }
            }
        }
        const float recall =
            static_cast<float>(correct) / static_cast<float>(nq_ * topk);
        return {recall, elapsed_us, violations};
    }

   private:
    const index_type& ivf_;
    const data_type& query_;
    const gt_type& gt_;
    const Args& args_;
    const AttributeMatrix<uint32_t>& base_attrs_;
    const AttributeMatrix<int32_t>& query_attrs_;
    const data_type& orig_base_;
    const data_type& orig_query_;
    size_t nq_;
    size_t search_k_;
    std::vector<PID> results_;
    std::vector<float> dists_;
    std::vector<PID> candidates_;
    std::vector<float> candidate_dists_;
};

std::vector<size_t> get_nprobes(
    SearchHarness& harness, const std::vector<size_t>& all_nprobes
) {
    float old_recall = 0;
    std::vector<size_t> nprobes;
    for (const size_t nprobe : all_nprobes) {
        nprobes.push_back(nprobe);
        const float recall = harness.run(nprobe).recall;
        if (recall > 0.997F || recall - old_recall < 1e-5F) {
            break;
        }
        std::cout << recall << '\t' << nprobe << std::endl << std::flush;
        old_recall = recall;
    }
    return nprobes;
}

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
    const size_t nq = static_cast<size_t>(query.rows());

    index_type ivf;
    ivf.load(args.index_file);

    data_type orig_base;
    data_type orig_query;
    if (args.rerank) {
        if (ivf.metric_type() != rabitqlib::METRIC_L2) {
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
            static_cast<size_t>(orig_base.rows()) < ivf.num_stored_points()) {
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
        search_k = std::min(topk * args.rerank_factor, ivf.num_points());
        if (search_k == 0) {
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
        if (base_attrs.rows < ivf.num_stored_points() || query_attrs.rows < nq) {
            throw std::runtime_error(
                "Attribute row count is smaller than the indexed dataset"
            );
        }
    }

    std::vector<size_t> all_nprobes;
    all_nprobes.push_back(5);
    for (size_t i = 10; i < 200; i += 10) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 200; i < 400; i += 40) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 400; i <= 1500; i += 100) {
        all_nprobes.push_back(i);
    }
    for (size_t i = 2000; i <= 4000; i += 500) {
        all_nprobes.push_back(i);
    }
    all_nprobes.push_back(6000);
    all_nprobes.push_back(10000);
    all_nprobes.push_back(15000);

    SearchHarness harness(
        ivf, query, gt, args, base_attrs, query_attrs, orig_base, orig_query,
        nq, search_k
    );
    const auto nprobes = get_nprobes(harness, all_nprobes);
    const size_t length = nprobes.size();

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));
    size_t predicate_violations = 0;

    for (size_t r = 0; r < test_round; ++r) {
        for (size_t l = 0; l < length; ++l) {
            const size_t nprobe = nprobes[l];
            if (nprobe > ivf.num_clusters()) {
                std::cout << "nprobe " << nprobe << " is larger than number of clusters, ";
                std::cout << "will use nprobe = num_cluster (" << ivf.num_clusters()
                          << ").\n";
            }
            const Outcome outcome = harness.run(nprobe);
            all_qps[r][l] = static_cast<float>(nq) / (outcome.elapsed_us / 1e6F);
            all_recall[r][l] = outcome.recall;
            predicate_violations += outcome.violations;
        }
    }

    const auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    const auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "nprobe\tQPS\trecall" << '\n';
    for (size_t i = 0; i < length; ++i) {
        std::cout << nprobes[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i]
                  << '\n';
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
