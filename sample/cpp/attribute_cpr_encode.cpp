// attribute_cpr_encode.cpp -- CPR (categorical disjoint-block) offline encoder.
//
// Mirrors attribute_apr_encode.cpp but uses the disjoint-block rotation
// (AprRotationPlan::make_cpr / rotate_batch_cpr). Each attribute owns a disjoint
// block of dimension pairs, so the encoded base and query share the exact
// distance-preserving property of APR while spreading the categorical phase
// codes across non-overlapping regions of the vector.
//
// Build: this target is added by sample/cpp/CMakeLists.txt (links rabitqlib::core).

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "attribute_rotation_cli.hpp"

int main(int argc, char* argv[]) {
    try {
        const part_sample::Args args =
            part_sample::parse_args(argc, argv, part_sample::EncoderKind::kCpr);
        const part_sample::RotationContext ctx =
            part_sample::prepare_rotation(args);

        std::filesystem::create_directories(args.out_dir);

        part_sample::AttributeMatrix<float> base_vecs =
            part_sample::load_vectors(
                args.base.c_str(), args.dim, ctx.base_attrs.rows
            );
        std::fprintf(
            stderr, "[cpr] rotating %zu base rows from %s\n", base_vecs.rows,
            args.base.c_str()
        );
        part_sample::rotate_vectors(args, ctx, base_vecs, true);
        const std::string base_output =
            part_sample::output_path(args, args.base_out);
        part_sample::write_fvecs(
            base_output.c_str(), base_vecs.data(), base_vecs.rows, args.dim
        );
        std::fprintf(stderr, "[cpr] wrote %s\n", base_output.c_str());

        part_sample::AttributeMatrix<float> query_vecs =
            part_sample::load_vectors(
                args.query.c_str(), args.dim, ctx.query_attrs.rows
            );
        std::fprintf(
            stderr, "[cpr] rotating %zu query rows from %s\n", query_vecs.rows,
            args.query.c_str()
        );
        part_sample::rotate_vectors(args, ctx, query_vecs, false);
        const std::string query_output =
            part_sample::output_path(args, args.query_out);
        part_sample::write_fvecs(
            query_output.c_str(), query_vecs.data(), query_vecs.rows, args.dim
        );
        std::fprintf(stderr, "[cpr] wrote %s\n", query_output.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "attribute_cpr_encode: %s\n", error.what());
        return 1;
    }
}
