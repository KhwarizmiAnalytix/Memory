"""Installable Memory artifacts and a relocatable CMake consumer interface."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_shared_library_info.bzl", "CcSharedLibraryInfo")

def _external_path(path):
    if path.startswith("../"):
        return path[3:]
    if path.startswith("external/"):
        return path[len("external/"):]
    if "/external/" in path:
        return path.split("/external/", 1)[1]
    return None

def _distribution_impl(ctx):
    outputs = {}
    windows = ctx.target_platform_has_constraint(ctx.attr._windows[platform_common.ConstraintValueInfo])
    main_name = "Memory.lib" if windows else "libMemory.a"
    runtime_name = None
    if ctx.attr.shared:
        library = ctx.attr.library[CcSharedLibraryInfo].linker_input.libraries[0]
        dynamic = library.resolved_symlink_dynamic_library or library.dynamic_library
        main_name = dynamic.basename
        outputs["lib/" + main_name] = dynamic
        runtime_name = main_name
        interface = library.resolved_symlink_interface_library or library.interface_library
        if windows and interface:
            outputs["lib/Memory.lib"] = interface
    else:
        files = ctx.attr.library[DefaultInfo].files.to_list()
        outputs["lib/" + main_name] = files[0]

    info = ctx.attr.implementation[CcInfo]
    link_libraries = []
    link_flags = []
    for linker_input in info.linking_context.linker_inputs.to_list():
        link_flags += linker_input.user_link_flags
        for library in linker_input.libraries:
            # Compiled objects are already merged into the static archive or
            # linked into the shared library. Prebuilt dependencies stay explicit.
            if library.objects or library.pic_objects:
                continue
            static = library.pic_static_library or library.static_library
            dynamic = library.resolved_symlink_dynamic_library or library.dynamic_library
            interface = library.resolved_symlink_interface_library or library.interface_library
            dependency = static or interface or dynamic
            if dependency:
                outputs["lib/" + dependency.basename] = dependency
                link_libraries.append("${_memory_prefix}/lib/" + dependency.basename)
            if dynamic:
                outputs["lib/" + dynamic.basename] = dynamic

    includes = ["${_memory_prefix}/include"]
    context = info.compilation_context
    for directory in context.includes.to_list() + context.system_includes.to_list() + context.quote_includes.to_list():
        path = _external_path(directory)
        if path:
            includes.append("${_memory_prefix}/include/memory_deps/" + path)
    for header in context.headers.to_list():
        path = _external_path(header.short_path)
        if path:
            outputs["include/memory_deps/" + path] = header
            includes.append("${_memory_prefix}/include/memory_deps/" + path.split("/")[0])

    files = []
    for destination, source in outputs.items():
        output = ctx.actions.declare_file("distribution/" + destination)
        ctx.actions.symlink(output = output, target_file = source)
        files.append(output)
    defines = [d for d in context.defines.to_list() if not d.endswith("_BUILDING_DLL")]
    config = ctx.actions.declare_file("distribution/lib/cmake/Memory/MemoryConfig.cmake")
    content = 'get_filename_component(_memory_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)\n'
    content += 'if(NOT TARGET Memory::Memory)\n'
    content += '  add_library(Memory::Memory %s IMPORTED)\n' % ("SHARED" if ctx.attr.shared else "STATIC")
    content += '  set_target_properties(Memory::Memory PROPERTIES\n'
    content += '    IMPORTED_LOCATION "${_memory_prefix}/lib/%s"\n' % main_name
    if windows and runtime_name:
        content += '    IMPORTED_IMPLIB "${_memory_prefix}/lib/Memory.lib"\n'
    content += '    INTERFACE_INCLUDE_DIRECTORIES "%s"\n' % ";".join(depset(includes).to_list())
    content += '    INTERFACE_COMPILE_DEFINITIONS "%s"\n' % ";".join(defines)
    # Framework flags are a pair; retaining them as link libraries preserves
    # their ordering on all supported CMake generators.
    content += '    INTERFACE_LINK_LIBRARIES "%s"\n' % ";".join(depset(link_libraries).to_list() + link_flags)
    content += '  )\nendif()\nunset(_memory_prefix)\n'
    ctx.actions.write(config, content)
    files.append(config)
    return [DefaultInfo(files = depset(files))]

memory_distribution = rule(
    implementation = _distribution_impl,
    attrs = {
        "library": attr.label(mandatory = True),
        "implementation": attr.label(mandatory = True, providers = [CcInfo]),
        "shared": attr.bool(default = False),
        "_windows": attr.label(default = "@platforms//os:windows"),
    },
)
