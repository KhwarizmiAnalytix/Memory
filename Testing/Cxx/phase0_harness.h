// Phase 0 measurement harness (plan §6.7, tasks 0.1/0.2/0.4).
//
// Header-only, dependency-free. Produces a JSON document with a manifest and raw
// samples. Sampling method: one sample = mean latency of `batch` back-to-back
// operations (batch chosen so a sample spans many timer ticks). Percentiles are
// nearest-rank over the samples; they describe batch means, NOT individual-op
// tails. The manifest records the timer resolution so this limit is explicit.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace phase0
{

using clock_t_ = std::chrono::steady_clock;

inline double now_ns()
{
    return std::chrono::duration<double, std::nano>(clock_t_::now().time_since_epoch()).count();
}

// Smallest observable tick of the clock, measured.
inline double timer_resolution_ns()
{
    double best = 1e18;
    for (int i = 0; i < 2000; ++i)
    {
        double a = now_ns();
        double b = now_ns();
        while (b == a)
        {
            b = now_ns();
        }
        best = std::min(best, b - a);
    }
    return best;
}

inline std::string shell_line(char const* cmd)
{
    std::string out;
#if defined(_WIN32)
    FILE* p = _popen(cmd, "r");
#else
    FILE* p = popen(cmd, "r");
#endif
    if (p == nullptr)
    {
        return "unknown";
    }
    char buf[256];
    while (std::fgets(buf, sizeof buf, p) != nullptr)
    {
        out += buf;
    }
#if defined(_WIN32)
    _pclose(p);
#else
    pclose(p);
#endif
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
    {
        out.pop_back();
    }
    return out.empty() ? "unknown" : out;
}

inline std::string json_escape(std::string const& s)
{
    std::string o;
    for (char c : s)
    {
        if (c == '"' || c == '\\')
        {
            o += '\\';
            o += c;
        }
        else if (static_cast<unsigned char>(c) < 0x20)
        {
            o += ' ';
        }
        else
        {
            o += c;
        }
    }
    return o;
}

inline double percentile(std::vector<double> sorted, double p)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    std::sort(sorted.begin(), sorted.end());
    auto rank = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(sorted.size())));
    rank      = std::min(std::max<size_t>(rank, 1), sorted.size());
    return sorted[rank - 1];
}

struct result
{
    std::string                        name;
    std::vector<std::pair<std::string, std::string>> params;  // string-valued
    std::string                        status{"ok"};  // "ok" or "skipped: <reason>"
    size_t                             batch{1};
    std::vector<double>                samples_ns_per_op;
    std::vector<std::pair<std::string, double>> counters;  // e.g. heap allocs, driver calls
};

class report
{
public:
    // repo: any path inside the source checkout (the process cwd is usually the build dir).
    explicit report(std::string suite, std::string build_note = {}, std::string const& repo = ".")
        : suite_(std::move(suite)), build_note_(std::move(build_note))
    {
        timer_res_ns_ = timer_resolution_ns();
        std::string const g = "git -C \"" + repo + "\" ";
        git_sha_   = shell_line((g + "rev-parse --short HEAD").c_str());
        git_dirty_ =
            shell_line((g + "status --porcelain --untracked-files=no").c_str()) == "unknown" ? "clean" : "dirty";
    }

    void note(std::string k, std::string v) { notes_.emplace_back(std::move(k), std::move(v)); }

    // Support-matrix keys (Testing/tools/support_matrix.py): backend = cpu|cuda|hip|metal,
    // evidence = compile|shim|hardware.
    void set_evidence(std::string backend, std::string evidence)
    {
        backend_  = std::move(backend);
        evidence_ = std::move(evidence);
    }
    void add(result r) { results_.push_back(std::move(r)); }

    // Compile-environment facts supplied by the caller's translation unit.
    void set_build(std::string compiler, bool ndebug)
    {
        compiler_ = std::move(compiler);
        ndebug_   = ndebug;
    }

    bool write(std::string const& path) const
    {
        std::ofstream f(path, std::ios::binary);
        if (!f)
        {
            return false;
        }
        f << "{\n  \"manifest\": {\n";
        f << "    \"suite\": \"" << json_escape(suite_) << "\",\n";
        f << "    \"backend\": \"" << backend_ << "\",\n";
        f << "    \"evidence\": \"" << evidence_ << "\",\n";
        f << "    \"git_sha\": \"" << git_sha_ << "\",\n";
        f << "    \"git_tree\": \"" << git_dirty_ << "\",\n";
        f << "    \"compiler\": \"" << json_escape(compiler_) << "\",\n";
        f << "    \"build\": \"" << (ndebug_ ? "Release (NDEBUG)" : "Debug (assertions on)") << "\",\n";
        f << "    \"hardware_threads\": " << std::thread::hardware_concurrency() << ",\n";
        f << "    \"timer_resolution_ns\": " << timer_res_ns_ << ",\n";
        f << "    \"sampling\": \"each sample = mean ns/op over `batch` ops; percentiles are "
             "nearest-rank over samples (batch means, not single-op tails)\",\n";
        f << "    \"evidence_level\": \"" << json_escape(build_note_) << "\",\n";
        f << "    \"notes\": {";
        for (size_t i = 0; i < notes_.size(); ++i)
        {
            f << (i ? ", " : "") << "\"" << json_escape(notes_[i].first) << "\": \""
              << json_escape(notes_[i].second) << "\"";
        }
        f << "}\n  },\n  \"results\": [\n";
        for (size_t i = 0; i < results_.size(); ++i)
        {
            auto const& r = results_[i];
            f << "    {\"name\": \"" << json_escape(r.name) << "\", \"status\": \""
              << json_escape(r.status) << "\", \"params\": {";
            for (size_t k = 0; k < r.params.size(); ++k)
            {
                f << (k ? ", " : "") << "\"" << json_escape(r.params[k].first) << "\": \""
                  << json_escape(r.params[k].second) << "\"";
            }
            f << "}, \"batch\": " << r.batch << ", \"n\": " << r.samples_ns_per_op.size()
              << ", \"p50_ns\": " << percentile(r.samples_ns_per_op, 50)
              << ", \"p95_ns\": " << percentile(r.samples_ns_per_op, 95)
              << ", \"p99_ns\": " << percentile(r.samples_ns_per_op, 99) << ", \"counters\": {";
            for (size_t k = 0; k < r.counters.size(); ++k)
            {
                f << (k ? ", " : "") << "\"" << json_escape(r.counters[k].first)
                  << "\": " << r.counters[k].second;
            }
            f << "}, \"samples_ns_per_op\": [";
            for (size_t k = 0; k < r.samples_ns_per_op.size(); ++k)
            {
                f << (k ? "," : "") << r.samples_ns_per_op[k];
            }
            f << "]}" << (i + 1 < results_.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
        return static_cast<bool>(f);
    }

private:
    std::string suite_;
    std::string build_note_;
    std::string git_sha_;
    std::string git_dirty_;
    std::string backend_{"unknown"};
    std::string evidence_{"unknown"};
    std::string compiler_{"unknown"};
    bool        ndebug_{false};
    double      timer_res_ns_{0};
    std::vector<std::pair<std::string, std::string>> notes_;
    std::vector<result>                              results_;
};

// Time `op` (one logical operation) with explicit warmup. `batch` ops per sample.
inline std::vector<double> measure(
    size_t warmup, size_t samples, size_t batch, std::function<void()> const& op)
{
    for (size_t i = 0; i < warmup; ++i)
    {
        op();
    }
    std::vector<double> out;
    out.reserve(samples);
    for (size_t s = 0; s < samples; ++s)
    {
        double const t0 = now_ns();
        for (size_t i = 0; i < batch; ++i)
        {
            op();
        }
        double const t1 = now_ns();
        out.push_back((t1 - t0) / static_cast<double>(batch));
    }
    return out;
}

// Batch size so one sample is >= ~64 timer ticks (keeps quantization < 2%).
inline size_t pick_batch(double resolution_ns, std::function<void()> const& op)
{
    double const t0 = now_ns();
    for (int i = 0; i < 64; ++i)
    {
        op();
    }
    double const per = std::max((now_ns() - t0) / 64.0, 1.0);
    auto b = static_cast<size_t>(std::ceil(64.0 * resolution_ns / per));
    return std::min<size_t>(std::max<size_t>(b, 1), 4096);
}

inline std::string arg_value(int argc, char** argv, std::string const& key, std::string def)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (key == argv[i])
        {
            return argv[i + 1];
        }
    }
    return def;
}

inline bool has_flag(int argc, char** argv, std::string const& key)
{
    for (int i = 1; i < argc; ++i)
    {
        if (key == argv[i])
        {
            return true;
        }
    }
    return false;
}

inline std::string compiler_string()
{
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#else
    return "unknown";
#endif
}

#ifdef NDEBUG
inline constexpr bool kNdebug = true;
#else
inline constexpr bool kNdebug = false;
#endif

}  // namespace phase0
