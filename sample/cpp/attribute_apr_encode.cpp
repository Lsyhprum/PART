#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/attribute/apr_rotation.hpp"
#include "rabitqlib/attribute/attr_io.hpp"
#include "rabitqlib/utils/io.hpp"

namespace {

using rabitqlib::attribute::AprConfig;
using rabitqlib::attribute::AprRotationPlan;
using rabitqlib::attribute::AttributeKind;
using rabitqlib::attribute::AttributeMatrix;
using rabitqlib::attribute::PredicateKind;
using rabitqlib::attribute::WildcardPolicy;
using rabitqlib::attribute::read_attr_bin;
using rabitqlib::attribute::read_salt_file;
using rabitqlib::attribute::write_fvecs;

struct Args {
    std::string base;
    std::string query;
    std::string base_attr;
    std::string query_attr;
    std::string out_dir = ".";
    std::string salt_file;
    std::string base_out = "apr_base.fvecs";
    std::string query_out = "apr_query.fvecs";
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
};

void usage(const char* program) {
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
}

size_t parse_size(const char* value, const char* option) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("Invalid value for ") + option);
    }
    return static_cast<size_t>(parsed);
}

double parse_range_l(const char* value) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
        throw std::invalid_argument("Invalid value for --range-l");
    }
    return parsed;
}

std::vector<AttributeKind> parse_layout(const std::string& text) {
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

Args parse_args(int argc, char* argv[]) {
    Args args;
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
        } else if (option == "--layout") {
            args.layout_text = value;
        } else if (option == "--meta-out") {
            args.meta_out = value;
        } else if (option == "--dim") {
            args.dim = parse_size(value, "--dim");
        } else if (option == "--attr-dim") {
            args.attr_dim = parse_size(value, "--attr-dim");
        } else if (option == "--threads") {
            args.threads = parse_size(value, "--threads");
        } else if (option == "--range-l") {
            args.range_l = parse_range_l(value);
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

    args.layout = args.layout_text.empty()
                      ? std::vector<AttributeKind>(args.attr_dim, AttributeKind::kEquality)
                      : parse_layout(args.layout_text);
    if (args.layout.size() != args.attr_dim) {
        throw std::invalid_argument(
            "--layout token count must equal --attr-dim"
        );
    }
    args.hapr = std::any_of(
        args.layout.begin(), args.layout.end(),
        [](AttributeKind kind) { return kind == AttributeKind::kRange; }
    );
    if (!args.hapr && args.range_l != 0.0) {
        throw std::invalid_argument(
            "--range-l requires a --layout containing at least one range token"
        );
    }
    return args;
}

size_t query_stride(const std::vector<AttributeKind>& layout) {
    size_t stride = 0;
    for (const AttributeKind kind : layout) {
        stride += kind == AttributeKind::kRange ? 2 : 1;
    }
    return stride;
}

// A single rotated probe uses the interval centre, so query attribute rows are
// wider than base rows: eq occupies one column, range occupies [lo, hi].
std::vector<double> expand_base_values(
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

std::vector<double> expand_query_values(
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

double infer_range_l(
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

void write_meta(const Args& args, double range_l, size_t qstride) {
    const std::string path =
        (std::filesystem::path(args.out_dir) / args.meta_out).string();
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot write HAPR meta file: " + path);
    }
    output << std::setprecision(17);
    output << "layout=" << args.layout_text << '\n';
    output << "range_l=" << range_l << '\n';
    if (args.use_standalone) {
        output << "standalone_seed=" << args.standalone_seed << '\n';
    } else {
        output << "salt_file=" << args.salt_file << '\n';
    }
    output << "dim=" << args.dim << '\n';
    output << "attr_dim=" << args.attr_dim << '\n';
    output << "qstride=" << qstride << '\n';
    output << "base_out=" << args.base_out << '\n';
    output << "query_out=" << args.query_out << '\n';
    if (!output) {
        throw std::runtime_error("Failed to write HAPR meta file: " + path);
    }
}

AttributeMatrix<float> load_vectors(
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

std::string output_path(const Args& args, const std::string& name) {
    return (std::filesystem::path(args.out_dir) / name).string();
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Args args = parse_args(argc, argv);
        const AttributeMatrix<uint32_t> base_attrs =
            read_attr_bin<uint32_t>(args.base_attr.c_str());
        const AttributeMatrix<int32_t> query_attrs =
            read_attr_bin<int32_t>(args.query_attr.c_str());
        if (base_attrs.cols != args.attr_dim) {
            throw std::runtime_error(
                "Base attribute column count does not match --attr-dim"
            );
        }
        const size_t qstride = args.hapr ? query_stride(args.layout) : args.attr_dim;
        if (query_attrs.cols != qstride) {
            throw std::runtime_error(
                "Query attribute column count does not match the layout stride"
            );
        }

        double range_l = 0.0;
        if (args.hapr) {
            range_l = args.range_l > 0.0
                          ? args.range_l
                          : infer_range_l(base_attrs, args.layout);
        }
        const AprConfig config{
            args.dim,
            args.attr_dim,
            args.hapr ? PredicateKind::kRange : PredicateKind::kEquality,
            WildcardPolicy::kReject,
            range_l
        };

        std::vector<uint64_t> salt;
        if (!args.use_standalone) {
            salt = read_salt_file(args.salt_file.c_str(), args.attr_dim, args.dim / 2);
        }
        const auto make_plan = [&]() -> AprRotationPlan {
            if (args.hapr) {
                if (args.use_standalone) {
                    return AprRotationPlan::make_hapr_standalone(
                        config, args.standalone_seed, args.layout.data(),
                        args.layout.size(), range_l
                    );
                }
                return AprRotationPlan::make_hapr(
                    config, salt.data(), salt.size(), args.layout.data(),
                    args.layout.size(), range_l
                );
            }
            if (args.use_standalone) {
                return AprRotationPlan::make_standalone(config, args.standalone_seed);
            }
            return AprRotationPlan::make(config, salt.data(), salt.size());
        };
        const AprRotationPlan plan = make_plan();

        std::vector<double> base_values;
        std::vector<double> query_values;
        if (args.hapr) {
            base_values = expand_base_values(base_attrs, args.layout);
            query_values = expand_query_values(query_attrs, args.layout);
            std::fprintf(
                stderr,
                "[hapr] layout=%s range_l=%.6g qstride=%zu dim=%zu attr_dim=%zu\n",
                args.layout_text.c_str(), range_l, qstride, args.dim, args.attr_dim
            );
        }

        std::filesystem::create_directories(args.out_dir);

        AttributeMatrix<float> base_vecs =
            load_vectors(args.base.c_str(), args.dim, base_attrs.rows);
        std::fprintf(
            stderr, "[apr] rotating %zu rows from %s\n", base_vecs.rows, args.base.c_str()
        );
        if (args.hapr) {
            rabitqlib::attribute::rotate_batch_hapr(
                plan, base_vecs.data(), base_vecs.rows, base_values.data(),
                base_vecs.data(), args.threads
            );
        } else {
            rabitqlib::attribute::rotate_batch(
                plan, base_vecs.data(), base_vecs.rows, base_attrs.data(),
                base_vecs.data(), args.threads
            );
        }
        const std::string base_output = output_path(args, args.base_out);
        write_fvecs(
            base_output.c_str(), base_vecs.data(), base_vecs.rows, args.dim
        );
        std::fprintf(stderr, "[apr] wrote %s\n", base_output.c_str());

        AttributeMatrix<float> query_vecs =
            load_vectors(args.query.c_str(), args.dim, query_attrs.rows);
        std::fprintf(
            stderr, "[apr] rotating %zu rows from %s\n", query_vecs.rows,
            args.query.c_str()
        );
        if (args.hapr) {
            rabitqlib::attribute::rotate_batch_hapr(
                plan, query_vecs.data(), query_vecs.rows, query_values.data(),
                query_vecs.data(), args.threads
            );
        } else {
            rabitqlib::attribute::rotate_batch(
                plan, query_vecs.data(), query_vecs.rows, query_attrs.data(),
                query_vecs.data(), args.threads
            );
        }
        const std::string query_output = output_path(args, args.query_out);
        write_fvecs(
            query_output.c_str(), query_vecs.data(), query_vecs.rows, args.dim
        );
        std::fprintf(stderr, "[apr] wrote %s\n", query_output.c_str());

        if (args.hapr) {
            write_meta(args, range_l, qstride);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "attribute_apr_encode: %s\n", error.what());
        return 1;
    }
}
