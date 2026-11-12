// attribute_spr_encode.cpp -- SPR (scaled-phase rotation) offline encoder.
//
// Mirrors attribute_apr_encode.cpp but selects the SPR encoder
// (AprRotationPlan::make_spr / rotate_batch_spr), which is the pure-range
// specialisation of HAPR from the PART paper (Algorithm III). Every attribute
// column is a numeric range column, so the rotation reuses HAPR's scaled-phase
// kernel and the metadata file records the all-range layout and range_l.
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
            part_sample::parse_args(argc, argv, part_sample::EncoderKind::kSpr);
        const part_sample::RotationContext ctx =
            part_sample::prepare_rotation(args);

        std::filesystem::create_directories(args.out_dir);

        part_sample::AttributeMatrix<float> base_vecs =
            part_sample::load_vectors(
                args.base.c_str(), args.dim, ctx.base_attrs.rows
            );
        std::fprintf(
            stderr, "[spr] rotating %zu base rows from %s\n", base_vecs.rows,
            args.base.c_str()
        );
        part_sample::rotate_vectors(args, ctx, base_vecs, true);
        const std::string base_output =
            part_sample::output_path(args, args.base_out);
        part_sample::write_fvecs(
            base_output.c_str(), base_vecs.data(), base_vecs.rows, args.dim
        );
        std::fprintf(stderr, "[spr] wrote %s\n", base_output.c_str());

        part_sample::AttributeMatrix<float> query_vecs =
            part_sample::load_vectors(
                args.query.c_str(), args.dim, ctx.query_attrs.rows
            );
        std::fprintf(
            stderr, "[spr] rotating %zu query rows from %s\n", query_vecs.rows,
            args.query.c_str()
        );
        part_sample::rotate_vectors(args, ctx, query_vecs, false);
        const std::string query_output =
            part_sample::output_path(args, args.query_out);
        part_sample::write_fvecs(
            query_output.c_str(), query_vecs.data(), query_vecs.rows, args.dim
        );
        std::fprintf(stderr, "[spr] wrote %s\n", query_output.c_str());

        if (ctx.hapr) {
            part_sample::write_meta(args, ctx);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "attribute_spr_encode: %s\n", error.what());
        return 1;
    }
}
