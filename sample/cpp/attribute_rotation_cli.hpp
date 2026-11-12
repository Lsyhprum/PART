#pragma once

// Shared CLI helpers for the offline attribute-rotation encoders and the
// one-shot IVF pipeline. Keeping argument validation, plan construction and
// batch rotation in one place guarantees that the standalone encoders and the
// in-process pipeline derive byte-identical rotated vectors from the same
// salt/seed; the numerical kernels themselves live in rabitqlib::attribute.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/apr_rotation.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/utils/io.hpp"

namespace part_sample {

using rabitqlib::attribute::AprConfig;
using rabitqlib::attribute::AprRotationPlan;
using rabitqlib::attribute::AttributeKind;
using rabitqlib::attribute::AttributeMatrix;
using rabitqlib::attribute::PredicateKind;
using rabitqlib::attribute::WildcardPolicy;
using rabitqlib::attribute::read_attr_bin;
using rabitqlib::attribute::read_salt_file;
using rabitqlib::attribute::write_fvecs;

enum class EncoderKind { kApr, kCpr, kHapr, kSpr };

struct Args {
    std::string base;
    std::string query;
    std::string base_attr;
    std::string query_attr;
    std::string out_dir = ".";
    std::string salt_file;
    std::string base_out;
    std::string query_out;
    std::string layout_text;
    std::vector<AttributeKind> layout;
    std::string meta_out = "hapr_meta.txt";
    double range_l = 0.0;
    size_t dim = 0;
    size_t attr_dim = 0;
    size_t threads = 0;
    uint64_t standalone_seed = 0;
    bool use_standalone = false;
    bool hapr = false;
    EncoderKind kind = EncoderKind::kApr;
};

inline void set_defaults(Args& args, EncoderKind kind) {
    args.kind = kind;
    if (kind == EncoderKind::kApr) {
        args.base_out = "apr_base.fvecs";
        args.query_out = "apr_query.fvecs";
    } else if (kind == EncoderKind::kCpr) {
        args.base_out = "cpr_base.fvecs";
        args.query_out = "cpr_query.fvecs";
    } else if (kind == EncoderKind::kHapr) {
        args.base_out = "hapr_base.fvecs";
        args.query_out = "hapr_query.fvecs";
    } else {
        args.base_out = "spr_base.fvecs";
        args.query_out = "spr_query.fvecs";
    }
}

inline void usage(const char* program, EncoderKind kind) {
    if (kind == EncoderKind::kApr) {
        std::fprintf(
            stderr,
            "Usage: %s --base F --query F --base-attr F --query-attr F --dim D "
            "--attr-dim A [--salt-file F | --standalone-seed N] [options]\n"
            "  --out-dir DIR          output directory (default: .)\n"
            "  --base-out NAME        rotated base file name (default: apr_base.fvecs)\n"
            "  --query-out NAME       rotated query file name (default: apr_query.fvecs)\n"
            "  --threads N            row-wise parallel workers (default: serial)\n"
            "  --layout LIST          per-attribute kinds, e.g. eq,range (default: all eq)\n"
            "  --range-l L            numeric value range; 0 infers from base (default: 0)\n"
            "  --meta-out NAME        HAPR parameter record (default: hapr_meta.txt)\n"
            "  -h, --help             show this message\n",
            program
        );
    } else {
        std::fprintf(
            stderr,
            "Usage: %s --base F --query F --base-attr F --query-attr F --dim D "
            "--attr-dim A [--salt-file F | --standalone-seed N] [options]\n"
            "  --out-dir DIR       output directory (default: .)\n"
            "  --base-out NAME     rotated base file name (default: cpr_base.fvecs)\n"
            "  --query-out NAME    rotated query file name (default: cpr_query.fvecs)\n"
            "  --threads N         row-wise parallel workers (default: serial)\n"
            "  -h, --help          show this message\n",
            program
        );
    }
}

inline size_t parse_size(const char* value, const char* option) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("Invalid value for ") + option);
    }
    return static_cast<size_t>(parsed);
}

inline double parse_range_l(const char* value) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
        throw std::invalid_argument("Invalid value for --range-l");
    }
    return parsed;
}

inline std::vector<AttributeKind> parse_layout(const std::string& text) {
    std::vector<AttributeKind> layout;
    size_t start = 0;
    while (true) {
        const size_t comma = text.find(',', start);
        const std::string token = text.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start
        );
        if (token.empty()) {
            throw std::invalid_argument("--layout contains an empty token");
        }
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
    if (layout.empty()) {
        throw std::invalid_argument("--layout must not be empty");
    }
    return layout;
}

inline size_t query_stride(const std::vector<AttributeKind>& layout) {
    size_t stride = 0;
    for (const AttributeKind kind : layout) {
        stride += kind == AttributeKind::kRange ? 2 : 1;
    }
    return stride;
}

inline Args parse_args(int argc, char* argv[], EncoderKind kind) {
    Args args;
    set_defaults(args, kind);
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "-h" || option == "--help") {
            usage(argv[0], kind);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + option);
        }
        const char* value = argv[++i];
        if (option == "--base") {
            args.base = value;
        } else if (option == "--query") {
            args.query = value;
        } else if (option == "--base-attr") {
            args.base_attr = value;
        } else if (option == "--query-attr") {
            args.query_attr = value;
        } else if (option == "--out-dir") {
            args.out_dir = value;
        } else if (option == "--salt-file") {
            args.salt_file = value;
        } else if (option == "--base-out") {
            args.base_out = value;
        } else if (option == "--query-out") {
            args.query_out = value;
        } else if (option == "--layout" || option == "--range-l" ||
                   option == "--meta-out") {
            if (kind == EncoderKind::kCpr) {
                throw std::invalid_argument("Unknown option: " + option);
            }
            if (kind == EncoderKind::kSpr && option == "--layout") {
                throw std::invalid_argument("Unknown option: " + option);
            }
            if (option == "--layout") {
                args.layout_text = value;
            } else if (option == "--range-l") {
                args.range_l = parse_range_l(value);
            } else {
                args.meta_out = value;
            }
        } else if (option == "--dim") {
            args.dim = parse_size(value, "--dim");
        } else if (option == "--attr-dim") {
            args.attr_dim = parse_size(value, "--attr-dim");
        } else if (option == "--threads") {
            args.threads = parse_size(value, "--threads");
        } else if (option == "--standalone-seed") {
            args.standalone_seed = static_cast<uint64_t>(parse_size(value, option.c_str()));
            args.use_standalone = true;
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }
    if (args.base.empty() || args.query.empty() || args.base_attr.empty() ||
        args.query_attr.empty()) {
        throw std::invalid_argument("--base/--query/--base-attr/--query-attr are required");
    }
    if (args.dim == 0 || args.dim % 2 != 0) {
        throw std::invalid_argument("--dim must be a positive even number");
    }
    if (args.attr_dim == 0) {
        throw std::invalid_argument("--attr-dim must be positive");
    }
    if ((!args.salt_file.empty()) == args.use_standalone) {
        throw std::invalid_argument(
            "Specify exactly one of --salt-file and --standalone-seed"
        );
    }

    if (kind == EncoderKind::kCpr) {
        // CPR has no layout; every column is an equality column.
    } else if (kind == EncoderKind::kSpr) {
        // SPR fixes the layout to all-range; --layout is rejected earlier.
        args.layout = std::vector<AttributeKind>(
            args.attr_dim, AttributeKind::kRange
        );
        args.hapr = true;
    } else {
        // kApr and kHapr share the layout machinery; kHapr is the explicit
        // name for the mixed/range path, kApr keeps the original behaviour.
        args.layout = args.layout_text.empty()
                          ? std::vector<AttributeKind>(
                                args.attr_dim, AttributeKind::kEquality
                            )
                          : parse_layout(args.layout_text);
        if (args.layout.size() != args.attr_dim) {
            throw std::invalid_argument("--layout token count must equal --attr-dim");
        }
        args.hapr = std::any_of(
            args.layout.begin(), args.layout.end(),
            [](AttributeKind attribute_kind) {
                return attribute_kind == AttributeKind::kRange;
            }
        );
        if (!args.hapr && args.range_l != 0.0) {
            throw std::invalid_argument(
                "--range-l requires a --layout containing at least one range token"
            );
        }
    }
    return args;
}

// A single rotated probe uses the interval centre, so query attribute rows are
// wider than base rows: eq occupies one column, range occupies [lo, hi].
inline std::vector<double> expand_base_values(
    const AttributeMatrix<uint32_t>& attrs,
    const std::vector<AttributeKind>& layout
) {
    std::vector<double> values(attrs.rows * layout.size());
    for (size_t i = 0; i < attrs.rows; ++i) {
        for (size_t a = 0; a < layout.size(); ++a) {
            values[i * layout.size() + a] = static_cast<double>(attrs(i, a));
        }
    }
    return values;
}

inline std::vector<double> expand_query_values(
    const AttributeMatrix<int32_t>& attrs,
    const std::vector<AttributeKind>& layout
) {
    std::vector<double> values(attrs.rows * layout.size());
    for (size_t i = 0; i < attrs.rows; ++i) {
        size_t col = 0;
        for (size_t a = 0; a < layout.size(); ++a) {
            if (layout[a] == AttributeKind::kEquality) {
                const int32_t target = attrs(i, col);
                if (target < 0) {
                    throw std::invalid_argument(
                        "Query equality targets must be non-negative; -1 wildcards "
                        "cannot be encoded by a single rotated probe"
                    );
                }
                values[i * layout.size() + a] = static_cast<double>(target);
                ++col;
            } else {
                const int32_t lo = attrs(i, col);
                const int32_t hi = attrs(i, col + 1);
                if (lo < 0 || hi < 0 || lo > hi) {
                    throw std::invalid_argument(
                        "Range bounds must satisfy 0 <= lo <= hi; -1 wildcards "
                        "cannot be encoded by a single rotated probe"
                    );
                }
                // Keep the .5 half-integer centre; truncation would change the
                // phase and break the soft-filter geometry.
                values[i * layout.size() + a] =
                    (static_cast<double>(lo) + static_cast<double>(hi)) / 2.0;
                col += 2;
            }
        }
    }
    return values;
}

inline double infer_range_l(
    const AttributeMatrix<uint32_t>& base_attrs,
    const std::vector<AttributeKind>& layout
) {
    bool found = false;
    uint32_t low = 0;
    uint32_t high = 0;
    for (size_t a = 0; a < layout.size(); ++a) {
        if (layout[a] != AttributeKind::kRange) {
            continue;
        }
        for (size_t i = 0; i < base_attrs.rows; ++i) {
            const uint32_t value = base_attrs(i, a);
            if (!found) {
                low = value;
                high = value;
                found = true;
            } else {
                low = std::min(low, value);
                high = std::max(high, value);
            }
        }
    }
    if (!found || base_attrs.rows == 0) {
        throw std::runtime_error("Cannot infer --range-l from an empty range column");
    }
    return static_cast<double>(high) - static_cast<double>(low) + 1.0;
}

struct RotationContext {
    AttributeMatrix<uint32_t> base_attrs;
    AttributeMatrix<int32_t> query_attrs;
    std::vector<double> base_values;
    std::vector<double> query_values;
    std::optional<AprRotationPlan> plan;
    std::vector<AttributeKind> layout;
    double range_l = 0.0;
    size_t qstride = 0;
    bool hapr = false;

    const AprRotationPlan& rotation() const { return *plan; }
    // A null layout tells the index gate that every column is equality.
    const AttributeKind* layout_ptr() const {
        return layout.empty() ? nullptr : layout.data();
    }
};

// Load and validate attributes, then build the rotation plan. The caller may
// reuse base_attrs/query_attrs as the predicate gate for a filtered search.
inline RotationContext prepare_rotation(const Args& args) {
    RotationContext ctx;
    ctx.base_attrs = read_attr_bin<uint32_t>(args.base_attr.c_str());
    ctx.query_attrs = read_attr_bin<int32_t>(args.query_attr.c_str());
    if (ctx.base_attrs.cols != args.attr_dim) {
        throw std::runtime_error(
            "Base attribute column count does not match --attr-dim"
        );
    }
    ctx.layout = args.layout;
    ctx.hapr = args.hapr;
    ctx.qstride = ctx.hapr ? query_stride(ctx.layout) : args.attr_dim;
    if (ctx.query_attrs.cols != ctx.qstride) {
        throw std::runtime_error(
            "Query attribute column count does not match the layout stride"
        );
    }

    if (ctx.hapr) {
        ctx.range_l = args.range_l > 0.0
                          ? args.range_l
                          : infer_range_l(ctx.base_attrs, ctx.layout);
    }
    const AprConfig config{
        args.dim,
        args.attr_dim,
        ctx.hapr ? PredicateKind::kRange : PredicateKind::kEquality,
        WildcardPolicy::kReject,
        ctx.range_l
    };

    std::vector<uint64_t> salt;
    if (!args.use_standalone) {
        salt = read_salt_file(args.salt_file.c_str(), args.attr_dim, args.dim / 2);
    }
    const auto make_plan = [&]() -> AprRotationPlan {
        if (args.kind == EncoderKind::kCpr) {
            if (args.use_standalone) {
                return AprRotationPlan::make_cpr_standalone(
                    config, args.standalone_seed
                );
            }
            return AprRotationPlan::make_cpr(config, salt.data(), salt.size());
        }
        if (args.kind == EncoderKind::kSpr) {
            if (args.use_standalone) {
                return AprRotationPlan::make_spr_standalone(
                    config, args.standalone_seed, ctx.range_l
                );
            }
            return AprRotationPlan::make_spr(
                config, salt.data(), salt.size(), ctx.range_l
            );
        }
        if (ctx.hapr) {
            if (args.use_standalone) {
                return AprRotationPlan::make_hapr_standalone(
                    config, args.standalone_seed, ctx.layout.data(),
                    ctx.layout.size(), ctx.range_l
                );
            }
            return AprRotationPlan::make_hapr(
                config, salt.data(), salt.size(), ctx.layout.data(),
                ctx.layout.size(), ctx.range_l
            );
        }
        if (args.use_standalone) {
            return AprRotationPlan::make_standalone(config, args.standalone_seed);
        }
        return AprRotationPlan::make(config, salt.data(), salt.size());
    };
    ctx.plan.emplace(make_plan());

    if (ctx.hapr) {
        ctx.base_values = expand_base_values(ctx.base_attrs, ctx.layout);
        ctx.query_values = expand_query_values(ctx.query_attrs, ctx.layout);
        const char* tag = args.kind == EncoderKind::kSpr ? "spr" : "hapr";
        std::fprintf(
            stderr, "[%s] layout=%s range_l=%.6g qstride=%zu dim=%zu attr_dim=%zu\n",
            tag, args.layout_text.c_str(), ctx.range_l, ctx.qstride, args.dim, args.attr_dim
        );
    }
    return ctx;
}

inline AttributeMatrix<float> load_vectors(
    const char* file, size_t dim, size_t expected_rows
) {
    AttributeMatrix<float> vectors;
    rabitqlib::load_vecs<float>(file, vectors);
    if (vectors.cols != dim) {
        throw std::runtime_error(
            std::string("Vector dimension does not match --dim: ") + file
        );
    }
    if (vectors.rows != expected_rows) {
        throw std::runtime_error(
            std::string("Attribute row count does not match vectors: ") + file
        );
    }
    return vectors;
}

inline std::string output_path(const Args& args, const std::string& name) {
    return (std::filesystem::path(args.out_dir) / name).string();
}

// In-place rotation: src and dst alias the same buffer, matching the standalone
// encoders so the float32 results are bit-for-bit identical.
inline void rotate_vectors(
    const Args& args,
    const RotationContext& ctx,
    AttributeMatrix<float>& vectors,
    bool is_base
) {
    float* data = vectors.data();
    if (args.kind == EncoderKind::kCpr) {
        if (is_base) {
            rabitqlib::attribute::rotate_batch_cpr(
                ctx.rotation(), data, vectors.rows, ctx.base_attrs.data(), data,
                args.threads
            );
        } else {
            rabitqlib::attribute::rotate_batch_cpr(
                ctx.rotation(), data, vectors.rows, ctx.query_attrs.data(), data,
                args.threads
            );
        }
        return;
    }
    if (args.kind == EncoderKind::kSpr) {
        const double* values =
            is_base ? ctx.base_values.data() : ctx.query_values.data();
        rabitqlib::attribute::rotate_batch_spr(
            ctx.rotation(), data, vectors.rows, values, data, args.threads
        );
        return;
    }
    if (ctx.hapr) {
        const double* values =
            is_base ? ctx.base_values.data() : ctx.query_values.data();
        rabitqlib::attribute::rotate_batch_hapr(
            ctx.rotation(), data, vectors.rows, values, data, args.threads
        );
    } else if (is_base) {
        rabitqlib::attribute::rotate_batch(
            ctx.rotation(), data, vectors.rows, ctx.base_attrs.data(), data,
            args.threads
        );
    } else {
        rabitqlib::attribute::rotate_batch(
            ctx.rotation(), data, vectors.rows, ctx.query_attrs.data(), data,
            args.threads
        );
    }
}

inline void write_meta(const Args& args, const RotationContext& ctx) {
    const std::string path = output_path(args, args.meta_out);
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot write HAPR meta file: " + path);
    }
    output << std::setprecision(17);
    output << "layout=" << args.layout_text << '\n';
    output << "range_l=" << ctx.range_l << '\n';
    if (args.use_standalone) {
        output << "standalone_seed=" << args.standalone_seed << '\n';
    } else {
        output << "salt_file=" << args.salt_file << '\n';
    }
    output << "dim=" << args.dim << '\n';
    output << "attr_dim=" << args.attr_dim << '\n';
    output << "qstride=" << ctx.qstride << '\n';
    output << "base_out=" << args.base_out << '\n';
    output << "query_out=" << args.query_out << '\n';
    if (!output) {
        throw std::runtime_error("Failed to write HAPR meta file: " + path);
    }
}

}  // namespace part_sample
