# 1. Tell pybind11 to use the modern FindPython module.
set(PYBIND11_FINDPYTHON ON)
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# 2. Locate Python, NumPy, and pybind11 from the active Python environment.
# Request module-development files without requiring libpython for embedding,
# which manylinux images do not provide.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

# 3. Define the shared pybind11 module.
pybind11_add_module(_rabitqlib
    rabitq_bindings.cpp
    hnsw_bindings.cpp
    ivf_bindings.cpp
    symqg_bindings.cpp
    kmeans_bindings.cpp
    quant_bindings.cpp
)

add_library(rabitqlib::python ALIAS _rabitqlib)

target_include_directories(_rabitqlib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
rabitq_configure_target(_rabitqlib)
# Release preparation keeps this in sync with the Python package version.
target_compile_definitions(_rabitqlib PRIVATE RABITQ_VERSION="${PROJECT_VERSION}")
# rabitqlib.quant results must not depend on the thread count or call site. GCC
# contracts a * b + c into FMA per compiled loop (the default -ffp-contract=fast on
# aarch64 or with -mfma), so the serial and OpenMP loops could round the inlined
# estimator differently. MSVC does not contract by default.
set_source_files_properties(quant_bindings.cpp PROPERTIES COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>"
)

# 4. Imported targets provide the matching Python, NumPy, and OpenMP settings.
target_link_libraries(_rabitqlib PRIVATE
    rabitqlib::index
    Python::NumPy
)

# Keep the OpenMP runtime discoverable in source installs and during wheel
# repair. Delocate copies it into the wheel and rewrites the loader path.
if(APPLE)
    set_property(TARGET _rabitqlib PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
    # CMake omits link directories inside the source/build tree from the
    # automatic install RPATH, including project-local Python environments.
    foreach(runtime IN LISTS OpenMP_CXX_LIBRARIES)
        get_filename_component(runtime_dir "${runtime}" DIRECTORY)
        set_property(TARGET _rabitqlib APPEND PROPERTY INSTALL_RPATH "${runtime_dir}")
    endforeach()
    install(FILES licenses/LLVM-OpenMP.txt DESTINATION rabitqlib/licenses)
endif()

# Linux release wheels bundle GNU OpenMP during auditwheel repair.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    install(FILES
        licenses/GNU-OpenMP.txt
        licenses/GPL-3.0.txt
        licenses/GCC-Runtime-Library-Exception.txt
        DESTINATION rabitqlib/licenses
    )
endif()

# 5. Map files to a 'rabitqlib' directory
# inside the target wheel, completely bypassing your local layout.
install(TARGETS _rabitqlib DESTINATION rabitqlib)
install(FILES __init__.py __init__.pyi py.typed quant.py quant.pyi DESTINATION rabitqlib)
