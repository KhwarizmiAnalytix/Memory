/*
 * XSigma: High-Performance Computational Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Contact: licensing@xsigma.co.uk
 */

// Phase 1.1 churn-crash reproduction.
//
// This benchmark removes the ->Iterations() caps from the cold and warm paths
// so Google Benchmark's own convergence drives iteration counts.  At
// --benchmark_min_time=0.05s --benchmark_repetitions=10 the cold path reaches
// ~250 iterations/rep and the warm path ~20 000; those counts, across multiple
// size variants and 10 reps each, reproduce the 30-40% crash rate documented
// in Docs/cpu_gpu_memory_review.md item 8 and CLAUDE.md.
//
// A Windows unhandled-exception filter is installed before benchmarks run (via
// static initialisation of churn_crash_handler_installer_).  On crash it writes
// churn_crash_report.txt and churn_crash.dmp in the binary's working directory
// (bin/).  The minidump can be analysed with llvm-symbolizer or WinDBG; the
// text file contains a DbgHelp stack trace that is useful even without a
// dedicated debugger.
//
// A baseline benchmark (BM_Churn_DirectMalloc) that calls cudaMalloc/cudaFree
// directly — without any caching-allocator code — is included as a control: if
// it also crashes at the same rate, the fault is in the CUDA driver/runtime
// rather than in the allocator.
//
// Run command:
//   bin\benchmark_memory_cudacachingallocatorchurn.exe \
//     --benchmark_min_time=0.05s --benchmark_repetitions=10

#include <benchmark/benchmark.h>

#if MEMORY_HAS_CUDA || MEMORY_HAS_HIP

#include <cstddef>
#include <vector>

#include "gpu/cuda_caching_allocator.h"
#include "gpu/device_guard.h"
#include "gpu/gpu_runtime.h"

// ---------------------------------------------------------------------------
// Windows crash handler — installed before any benchmark runs via static init
// ---------------------------------------------------------------------------
#if defined(_WIN32)

// clang-format off
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
// clang-format on

#pragma comment(lib, "dbghelp.lib")

#include <cstdio>
#include <ctime>

static LONG WINAPI churn_crash_handler(EXCEPTION_POINTERS* ep) noexcept
{
    // Re-entry guard: a crash inside the handler must not recurse.
    static volatile LONG guard = 0;
    if (InterlockedCompareExchange(&guard, 1, 0) != 0)
    {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    FILE* report = nullptr;
    fopen_s(&report, "churn_crash_report.txt", "w");
    if (report == nullptr)
    {
        report = stderr;
    }

    {
        time_t t = {};
        time(&t);
        fprintf(report, "=== Churn Crash Report ===\n");
        fprintf(report, "Time           : %s", ctime(&t));
        fprintf(
            report,
            "Exception code : 0x%08lX\n",
            ep->ExceptionRecord->ExceptionCode);
        fprintf(
            report,
            "Exception addr : %p\n",
            ep->ExceptionRecord->ExceptionAddress);
        if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
            ep->ExceptionRecord->NumberParameters >= 2)
        {
            fprintf(
                report,
                "Access type    : %s\n",
                ep->ExceptionRecord->ExceptionInformation[0] == 1 ? "write" : "read");
            fprintf(
                report,
                "Fault address  : 0x%llx\n",
                static_cast<unsigned long long>(ep->ExceptionRecord->ExceptionInformation[1]));
        }
    }

    // Initialise DbgHelp with deferred symbol loading and source-line info.
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    if (SymInitialize(proc, nullptr, TRUE))
    {
        // Walk the stack from the faulting context.
        CONTEXT ctx = *ep->ContextRecord;
        STACKFRAME64 frame{};
        frame.AddrPC.Offset    = ctx.Rip;
        frame.AddrPC.Mode      = AddrModeFlat;
        frame.AddrStack.Offset = ctx.Rsp;
        frame.AddrStack.Mode   = AddrModeFlat;
        frame.AddrFrame.Offset = ctx.Rbp;
        frame.AddrFrame.Mode   = AddrModeFlat;

        alignas(SYMBOL_INFO) char sym_buf[sizeof(SYMBOL_INFO) + 512];
        auto* sym          = reinterpret_cast<PSYMBOL_INFO>(sym_buf);
        sym->SizeOfStruct  = sizeof(SYMBOL_INFO);
        sym->MaxNameLen    = 511;

        fprintf(report, "\nStack trace (innermost first):\n");
        for (int i = 0; i < 64; ++i)
        {
            BOOL ok = StackWalk64(
                IMAGE_FILE_MACHINE_AMD64,
                proc,
                GetCurrentThread(),
                &frame,
                &ctx,
                nullptr,
                SymFunctionTableAccess64,
                SymGetModuleBase64,
                nullptr);
            if (!ok || frame.AddrPC.Offset == 0)
            {
                break;
            }

            DWORD64      disp64 = 0;
            DWORD        disp   = 0;
            IMAGEHLP_LINE64 line{};
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

            if (SymFromAddr(proc, frame.AddrPC.Offset, &disp64, sym))
            {
                if (SymGetLineFromAddr64(proc, frame.AddrPC.Offset, &disp, &line))
                {
                    fprintf(
                        report,
                        "  #%02d  %s+%llu  [%s:%lu]  (0x%016llx)\n",
                        i,
                        sym->Name,
                        static_cast<unsigned long long>(disp64),
                        line.FileName,
                        static_cast<unsigned long>(line.LineNumber),
                        static_cast<unsigned long long>(frame.AddrPC.Offset));
                }
                else
                {
                    fprintf(
                        report,
                        "  #%02d  %s+%llu  (0x%016llx)\n",
                        i,
                        sym->Name,
                        static_cast<unsigned long long>(disp64),
                        static_cast<unsigned long long>(frame.AddrPC.Offset));
                }
            }
            else
            {
                fprintf(
                    report,
                    "  #%02d  0x%016llx  (no symbol)\n",
                    i,
                    static_cast<unsigned long long>(frame.AddrPC.Offset));
            }
        }
        SymCleanup(proc);
    }
    else
    {
        fprintf(report, "\n[DbgHelp init failed — no stack trace]\n");
    }

    // Write a full minidump alongside the text report for WinDBG / llvm-symbolizer.
    HANDLE dmpfile = CreateFileA(
        "churn_crash.dmp",
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (dmpfile != INVALID_HANDLE_VALUE)
    {
        MINIDUMP_EXCEPTION_INFORMATION mdei{};
        mdei.ThreadId          = GetCurrentThreadId();
        mdei.ExceptionPointers = ep;
        mdei.ClientPointers    = FALSE;

        const MINIDUMP_TYPE dump_type = static_cast<MINIDUMP_TYPE>(
            MiniDumpWithFullMemory |
            MiniDumpWithModuleHeaders |
            MiniDumpWithHandleData);

        if (MiniDumpWriteDump(
                proc,
                GetCurrentProcessId(),
                dmpfile,
                dump_type,
                &mdei,
                nullptr,
                nullptr))
        {
            fprintf(report, "\nMinidump: churn_crash.dmp\n");
        }
        else
        {
            fprintf(report, "\n[MiniDumpWriteDump failed: 0x%08lX]\n", GetLastError());
        }
        CloseHandle(dmpfile);
    }

    fprintf(report, "=== End of Crash Report ===\n");
    if (report != stderr)
    {
        fclose(report);
    }

    return EXCEPTION_EXECUTE_HANDLER;
}

namespace
{
// Static-init installs the handler before benchmark::Initialize() runs.
struct churn_crash_handler_installer
{
    churn_crash_handler_installer() noexcept
    {
        SetUnhandledExceptionFilter(churn_crash_handler);
    }
};
}  // namespace
static const churn_crash_handler_installer churn_crash_handler_installer_;

#endif  // _WIN32

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace memory::benchmarks
{
namespace
{

bool cuda_device_available()
{
    int               n   = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    return err == cudaSuccess && n > 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// BM_Churn_ColdAllocFree
//   Identical to BM_Cuda_ColdAllocFree in BenchmarkCudaCachingAllocator.cpp
//   except the ->Iterations() cap is removed.  Each new benchmark-function
//   invocation (size variant × repetition) creates a fresh allocator, does the
//   cold-path loop, then destroys it — the construction/destruction cycle that
//   triggers the crash.
// ---------------------------------------------------------------------------
static void bm_churn_cold_alloc_free(benchmark::State& state)
{
    if (!cuda_device_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const std::size_t           size = static_cast<std::size_t>(state.range(0));
    gpu::cuda_caching_allocator allocator(0);

    for (auto _ : state)
    {
        allocator.empty_cache();
        void* ptr = allocator.allocate(size);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr, size);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
// NO ->Iterations() cap — this is intentional. Benchmark convergence drives
// the count and produces the volume that triggers the documented crash.
BENCHMARK(bm_churn_cold_alloc_free)
    ->Name("BM_Churn_ColdAllocFree")
    ->Range(4096, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_Churn_WarmAllocFree
//   Warm-path without iteration cap, for completeness.
// ---------------------------------------------------------------------------
static void bm_churn_warm_alloc_free(benchmark::State& state)
{
    if (!cuda_device_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const std::size_t           size = static_cast<std::size_t>(state.range(0));
    gpu::cuda_caching_allocator allocator(0);

    for (auto _ : state)
    {
        void* ptr = allocator.allocate(size);
        benchmark::DoNotOptimize(ptr);
        allocator.deallocate(ptr, size);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(bm_churn_warm_alloc_free)
    ->Name("BM_Churn_WarmAllocFree")
    ->Range(4096, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_Churn_DirectMalloc  (control)
//   Raw cudaMalloc/cudaFree without any caching-allocator code.  If this
//   benchmark also crashes at a similar rate, the fault is in the CUDA
//   driver/runtime, not in the caching allocator.
// ---------------------------------------------------------------------------
static void bm_churn_direct_malloc(benchmark::State& state)
{
    if (!cuda_device_available())
    {
        state.SkipWithError("No GPU device available");
        return;
    }
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
        void*             ptr = nullptr;
        const cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess)
        {
            state.SkipWithError("cudaMalloc failed");
            return;
        }
        benchmark::DoNotOptimize(ptr);
        cudaFree(ptr);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * size));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(bm_churn_direct_malloc)
    ->Name("BM_Churn_DirectMalloc")
    ->Range(4096, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

}  // namespace memory::benchmarks

#endif  // MEMORY_HAS_CUDA || MEMORY_HAS_HIP
