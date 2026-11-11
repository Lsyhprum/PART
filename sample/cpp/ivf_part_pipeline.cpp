// ivf_part_pipeline.cpp -- one-shot PART IVF build.
//
// Replaces the shell-level "encode -> cluster -> index" split: attribute
// rotation, centroid training, point routing/quantization and index saving all
// happen in one process, with no centroids/cluster-ids intermediate files.
//
// The index is built with IVF::train + IVF::add(nullptr) instead of
// IVF::construct, so centroids come from the library's own RaBitQKMeans /
// QGKMeans and every point is routed to its nearest centroid by the Flat
// initializer. Querying (including full nprobe/QPS/recall sweeps and rerank)
// remains the job of ivf_rabitq_querying; --gt enables a lightweight smoke
// check in this process.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "attribute_rotation_cli.hpp"
#include "rabitqlib/attribute/attr_filter.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/training.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/io.hpp"

namespace {

using rabitqlib::PID;
using part_sample::Args;
using part_sample::EncoderKind;

struct PipelineArgs {
    Args encoding;
    std::string index_out = "part_ivf.index";
    std::string gt_file;
    size_t num_clusters = 1024;
    size_t bits = 8;
    size_t nprobe = 0;   // 0 -> min(num_clusters, 16)
    size_t topk = 100;
    uint32_t index_seed = 42;
    uint32_t cluster_seed = 1234;
    std::string metric_text = "l2";
    bool layout_explicit = false;
};

void usage(const char* program) {
    std::fprintf(
        stderr,
        "Usage: %s --base F --query F --base-attr F --query-attr F --dim D "
        "--attr-dim A [--salt-file F | --standalone-seed N] [options]\n"
        "Build a rotated IVF-RaBitQ index in one process (no clustering files).\n"
        "  --encoder apr|cpr       rotation protocol (default: apr)\n"
        "  --out-dir DIR           output directory (default: .)\n"
        "  --index-out NAME        index file name (default: part_ivf.index)\n"
        "  --query-out NAME        rotated query file name (default: part_query.fvecs)\n"
        "  --num-clusters K        IVF centroid count (default: 1024)\n"
        "  --bits B                total quantization bits, 1-9 or 32 (default: 8)\n"
        "  --metric l2|ip          distance metric (default: l2)\n"
        "  --index-seed N          IVF rotator seed (default: 42)\n"
        "  --cluster-seed N        k-means seed (default: 1234)\n"
        "  --layout LIST           APR per-attribute kinds, e.g. eq,range (default: all eq)\n"
        "  --range-l L             numeric value range; 0 infers from base (default: 0)\n"
        "  --meta-out NAME         HAPR parameter record (default: hapr_meta.txt)\n"
        "  --threads N             workers for rotation, training, routing (default: library default)\n"
        "  --gt F                  optional ground-truth .ivecs for a filtered-search smoke test\n"
        "  --nprobe N              smoke-test nprobe (default: min(K, 16))\n"
        "  --topk N                smoke-test top-K (default: 100, capped by GT columns)\n"
        "  -h, --help              show this message\n",
        program
    );
}

size_t parse_u32(const char* value, const char* option) {
    const size_t parsed = part_sample::parse_size(value, option);
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(std::string("Value out of range for ") + option);
    }
    return parsed;
}

PipelineArgs parse_args(int argc, char* argv[]) {
    PipelineArgs result;
    Args& enc = result.encoding;
    part_sample::set_defaults(enc, EncoderKind::kApr);
    enc.query_out = "part_query.fvecs";
    enc.base_out = "(in-memory)";

    std::string encoder_text = "apr";
    std::string layout_text;
    double range_l = 0.0;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "-h" || option == "--help") {
            usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + option);
        }
        const char* value = argv[++i];
        if (option == "--base") {
            enc.base = value;
        } else if (option == "--query") {
            enc.query = value;
        } else if (option == "--base-attr") {
            enc.base_attr = value;
        } else if (option == "--query-attr") {
            enc.query_attr = value;
        } else if (option == "--out-dir") {
            enc.out_dir = value;
        } else if (option == "--salt-file") {
            enc.salt_file = value;
        } else if (option == "--index-out") {
            result.index_out = value;
        } else if (option == "--query-out") {
            enc.query_out = value;
        } else if (option == "--encoder") {
            encoder_text = value;
        } else if (option == "--gt") {
            result.gt_file = value;
        } else if (option == "--metric") {
            result.metric_text = value;
        } else if (option == "--layout") {
            layout_text = value;
            result.layout_explicit = true;
        } else if (option == "--meta-out") {
            enc.meta_out = value;
        } else if (option == "--dim") {
            enc.dim = part_sample::parse_size(value, "--dim");
        } else if (option == "--attr-dim") {
            enc.attr_dim = part_sample::parse_size(value, "--attr-dim");
        } else if (option == "--threads") {
            enc.threads = parse_u32(value, "--threads");
        } else if (option == "--range-l") {
            range_l = part_sample::parse_range_l(value);
        } else if (option == "--standalone-seed") {
            enc.standalone_seed = static_cast<uint64_t>(
                part_sample::parse_size(value, option.c_str())
            );
            enc.use_standalone = true;
        } else if (option == "--num-clusters") {
            result.num_clusters = part_sample::parse_size(value, "--num-clusters");
        } else if (option == "--bits") {
            result.bits = part_sample::parse_size(value, "--bits");
        } else if (option == "--nprobe") {
            result.nprobe = part_sample::parse_size(value, "--nprobe");
        } else if (option == "--topk") {
            result.topk = part_sample::parse_size(value, "--topk");
        } else if (option == "--index-seed") {
            result.index_seed = static_cast<uint32_t>(parse_u32(value, "--index-seed"));
        } else if (option == "--cluster-seed") {
            result.cluster_seed =
                static_cast<uint32_t>(parse_u32(value, "--cluster-seed"));
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }

    if (enc.base.empty() || enc.query.empty() || enc.base_attr.empty() ||
        enc.query_attr.empty()) {
        throw std::invalid_argument("--base/--query/--base-attr/--query-attr are required");
    }
    if (enc.dim == 0 || enc.dim % 2 != 0) {
        throw std::invalid_argument("--dim must be a positive even number");
    }
    if (enc.dim < 64) {
        throw std::invalid_argument(
            "--dim must be at least 64: the IVF FhtKacRotator cannot index smaller vectors"
        );
    }
    if (enc.attr_dim == 0) {
        throw std::invalid_argument("--attr-dim must be positive");
    }
    if ((!enc.salt_file.empty()) == enc.use_standalone) {
        throw std::invalid_argument(
            "Specify exactly one of --salt-file and --standalone-seed"
        );
    }
    if (encoder_text == "apr") {
        enc.kind = EncoderKind::kApr;
    } else if (encoder_text == "cpr") {
        enc.kind = EncoderKind::kCpr;
    } else {
        throw std::invalid_argument("--encoder must be apr or cpr");
    }
    if (enc.kind == EncoderKind::kCpr && (result.layout_explicit || range_l != 0.0)) {
        throw std::invalid_argument(
            "--layout / --range-l are only supported by --encoder apr"
        );
    }
    if (result.metric_text != "l2" && result.metric_text != "ip") {
        throw std::invalid_argument("--metric must be l2 or ip");
    }
    if (result.num_clusters == 0) {
        throw std::invalid_argument("--num-clusters must be positive");
    }
    if ((result.bits < 1 || result.bits > 9) && result.bits != 32) {
        throw std::invalid_argument("--bits must be in [1, 9] or 32");
    }
    if (result.topk == 0) {
        throw std::invalid_argument("--topk must be positive");
    }

    enc.layout_text = layout_text;
    enc.range_l = range_l;
    enc.layout = layout_text.empty()
                     ? std::vector<rabitqlib::attribute::AttributeKind>(
                           enc.attr_dim, rabitqlib::attribute::AttributeKind::kEquality
                       )
                     : part_sample::parse_layout(layout_text);
    if (enc.layout.size() != enc.attr_dim) {
        throw std::invalid_argument("--layout token count must equal --attr-dim");
    }
    enc.hapr = std::any_of(
        enc.layout.begin(), enc.layout.end(),
        [](rabitqlib::attribute::AttributeKind kind) {
            return kind == rabitqlib::attribute::AttributeKind::kRange;
        }
    );
    if (!enc.hapr && range_l != 0.0) {
        throw std::invalid_argument(
            "--range-l requires a --layout containing at least one range token"
        );
    }
    if (enc.kind == EncoderKind::kCpr) {
        enc.layout.clear();
        enc.hapr = false;
    }
    return result;
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void run_smoke(
    const rabitqlib::ivf::IVF& index,
    const PipelineArgs& args,
    const part_sample::RotationContext& ctx,
    const part_sample::AttributeMatrix<float>& query
) {
    rabitqlib::RowMajorArray<uint32_t> gt;
    rabitqlib::load_vecs<uint32_t>(args.gt_file.c_str(), gt);
    const size_t gt_cols = static_cast<size_t>(gt.cols());
    if (static_cast<size_t>(gt.rows()) != query.rows) {
        throw std::runtime_error("Ground-truth row count does not match --query");
    }
    const size_t k = std::min(args.topk, gt_cols);
    if (k == 0) {
        throw std::runtime_error("Ground truth has no columns");
    }
    const size_t nprobe = args.nprobe == 0
                              ? std::min<size_t>(args.num_clusters, 16)
                              : args.nprobe;
    const size_t num_threads = args.encoding.threads == 0 ? 1 : args.encoding.threads;

    std::vector<PID> ids(query.rows * k, rabitqlib::kPidMax);
    std::vector<float> distances(query.rows * k, 0.0F);
    const auto start = Clock::now();
    if (args.layout_explicit) {
        index.search_batch(
            query.data(), query.rows, k, nprobe, ids.data(), distances.data(),
            ctx.base_attrs.data(), ctx.query_attrs.data(), args.encoding.attr_dim,
            ctx.layout_ptr(), std::nullopt, num_threads
        );
    } else {
        index.search_batch(
            query.data(), query.rows, k, nprobe, ids.data(), distances.data(),
            ctx.base_attrs.data(), ctx.query_attrs.data(), args.encoding.attr_dim,
            std::nullopt, num_threads
        );
    }
    const double search_ms = elapsed_ms(start);

    size_t correct = 0;
    size_t violations = 0;
    const rabitqlib::attribute::AttributeKind* layout =
        args.layout_explicit ? ctx.layout_ptr() : nullptr;
    for (size_t z = 0; z < query.rows; ++z) {
        for (size_t y = 0; y < k; ++y) {
            const PID id = ids[z * k + y];
            if (id == rabitqlib::kPidMax) {
                continue;
            }
            if (!rabitqlib::attribute::predicate_matches(
                    ctx.base_attrs.data() +
                        static_cast<size_t>(id) * args.encoding.attr_dim,
                    ctx.query_attrs.data() + z * ctx.qstride,
                    args.encoding.attr_dim, layout)) {
                ++violations;
            }
            for (size_t g = 0; g < gt_cols; ++g) {
                if (gt.data()[z * gt_cols + g] == id) {
                    ++correct;
                    break;
                }
            }
        }
    }
    const double recall = static_cast<double>(correct) /
                          static_cast<double>(query.rows * k);
    std::fprintf(
        stderr,
        "[smoke] k=%zu nprobe=%zu recall=%.4f predicate_violations=%zu "
        "search_ms=%.1f\n",
        k, nprobe, recall, violations, search_ms
    );
}

int run(int argc, char* argv[]) {
    const PipelineArgs args = parse_args(argc, argv);
    const Args& enc = args.encoding;
    const part_sample::RotationContext ctx = part_sample::prepare_rotation(enc);

    part_sample::AttributeMatrix<float> base =
        part_sample::load_vectors(enc.base.c_str(), enc.dim, ctx.base_attrs.rows);
    if (args.num_clusters > base.rows) {
        throw std::invalid_argument(
            "--num-clusters must not exceed the number of base vectors"
        );
    }
    part_sample::AttributeMatrix<float> query =
        part_sample::load_vectors(enc.query.c_str(), enc.dim, ctx.query_attrs.rows);

    std::fprintf(
        stderr,
        "[pipeline] encoder=%s dim=%zu attr_dim=%zu n=%zu nq=%zu clusters=%zu "
        "bits=%zu metric=%s\n",
        enc.kind == EncoderKind::kApr ? "apr" : "cpr", enc.dim, enc.attr_dim,
        base.rows, query.rows, args.num_clusters, args.bits,
        args.metric_text.c_str()
    );

    auto rotate_start = Clock::now();
    part_sample::rotate_vectors(enc, ctx, base, true);
    part_sample::rotate_vectors(enc, ctx, query, false);
    std::fprintf(
        stderr, "[pipeline] rotated base/query in %.1f ms\n",
        elapsed_ms(rotate_start)
    );

    const rabitqlib::MetricType metric = args.metric_text == "ip"
                                             ? rabitqlib::METRIC_IP
                                             : rabitqlib::METRIC_L2;
    rabitqlib::ivf::IVF index(
        enc.dim, args.num_clusters, args.bits, metric, args.index_seed,
        rabitqlib::RotatorType::FhtKacRotator,
        rabitqlib::ivf::InitializerType::Flat
    );
    auto cp = rabitqlib::default_training_parameters(metric, args.cluster_seed);
    if (enc.threads > 0) {
        cp.num_threads = static_cast<uint32_t>(enc.threads);
    }
    std::fprintf(
        stderr,
        "[pipeline] training %zu centroids (niter=%zu sample<=%zu/centroid "
        "cluster_seed=%u threads=%u)\n",
        args.num_clusters, cp.niter, cp.max_points_per_centroid,
        args.cluster_seed, cp.num_threads
    );
    auto train_start = Clock::now();
    index.train(base.rows, base.data(), cp);
    std::fprintf(
        stderr, "[pipeline] trained %zu centroids in %.1f ms\n",
        args.num_clusters, elapsed_ms(train_start)
    );

    const size_t num_threads = enc.threads == 0
                                   ? std::numeric_limits<size_t>::max()
                                   : enc.threads;
    auto add_start = Clock::now();
    index.add(base.data(), base.rows, nullptr, false, num_threads);
    std::fprintf(
        stderr, "[pipeline] added %zu points (exact Flat routing) in %.1f ms\n",
        base.rows, elapsed_ms(add_start)
    );

    std::filesystem::create_directories(enc.out_dir);
    const std::string index_path = part_sample::output_path(enc, args.index_out);
    index.save(index_path.c_str());
    std::fprintf(stderr, "[pipeline] saved index -> %s\n", index_path.c_str());

    const std::string query_path = part_sample::output_path(enc, enc.query_out);
    part_sample::write_fvecs(
        query_path.c_str(), query.data(), query.rows, enc.dim
    );
    std::fprintf(stderr, "[pipeline] wrote rotated query -> %s\n", query_path.c_str());
    if (ctx.hapr) {
        part_sample::write_meta(enc, ctx);
    }

    if (!args.gt_file.empty()) {
        run_smoke(index, args, ctx, query);
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ivf_part_pipeline: %s\n", error.what());
        return 1;
    }
}
