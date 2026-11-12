# APR offline encoder only needs the core library (no index build required).
add_executable(attribute_apr_encode attribute_apr_encode.cpp)
target_link_libraries(attribute_apr_encode PRIVATE rabitqlib::core)
rabitq_configure_target(attribute_apr_encode)
set_target_properties(attribute_apr_encode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# CPR (categorical disjoint-block) offline encoder, same dependency as APR.
add_executable(attribute_cpr_encode attribute_cpr_encode.cpp)
target_link_libraries(attribute_cpr_encode PRIVATE rabitqlib::core)
rabitq_configure_target(attribute_cpr_encode)
set_target_properties(attribute_cpr_encode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# SPR (scaled-phase rotation) offline encoder, same dependency as APR.
add_executable(attribute_spr_encode attribute_spr_encode.cpp)
target_link_libraries(attribute_spr_encode PRIVATE rabitqlib::core)
rabitq_configure_target(attribute_spr_encode)
set_target_properties(attribute_spr_encode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(NOT RABITQ_BUILD_INDEX)
    return()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(symqg_indexing symqg_indexing.cpp)
add_executable(symqg_querying symqg_querying.cpp)
add_executable(rabitqkmeans rabitqkmeans.cpp)

add_executable(ivf_rabitq_indexing ivf_rabitq_indexing.cpp)
add_executable(ivf_rabitq_querying ivf_rabitq_querying.cpp)

# One-shot PART pipeline: rotation + centroid training + IVF build in one process.
add_executable(ivf_part_pipeline ivf_part_pipeline.cpp)



foreach(RABITQ_SAMPLE_TARGET
    symqg_indexing
    symqg_querying
    rabitqkmeans
    ivf_rabitq_indexing
    ivf_rabitq_querying
    ivf_part_pipeline
)
    target_link_libraries(${RABITQ_SAMPLE_TARGET} PRIVATE rabitqlib::index)
    rabitq_configure_target(${RABITQ_SAMPLE_TARGET})
endforeach()
