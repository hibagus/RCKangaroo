#include "rckangaroo/host_topology.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#include <Windows.h>
#else
#include <cerrno>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#endif

namespace rckangaroo {
namespace {

bool ParseCpu(std::string_view text, unsigned& cpu)
{
    if (text.empty()) {
        return false;
    }
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, cpu, 10);
    return result.ec == std::errc{} && result.ptr == end;
}

std::string ReadFirstLine(const std::string& path)
{
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    return line;
}

bool IsPciBusId(std::string_view value)
{
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return std::isxdigit(character) != 0 || character == ':' ||
                      character == '.';
           });
}

std::vector<unsigned> CurrentProcessCpus()
{
    std::vector<unsigned> result;
#ifdef _WIN32
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    for (unsigned cpu = 0; cpu < info.dwNumberOfProcessors; ++cpu) {
        result.push_back(cpu);
    }
#else
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return result;
    }
    for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(static_cast<int>(cpu), &allowed)) {
            result.push_back(cpu);
        }
    }
#endif
    return result;
}

std::vector<unsigned> IntersectCpus(const std::vector<unsigned>& preferred,
                                    const std::vector<unsigned>& allowed)
{
    std::vector<unsigned> result;
    std::set_intersection(preferred.begin(), preferred.end(),
                          allowed.begin(), allowed.end(),
                          std::back_inserter(result));
    return result;
}

} // namespace

bool ParseCpuList(std::string_view text,
                  std::vector<unsigned>& cpus,
                  std::string& error)
{
    cpus.clear();
    error.clear();
    if (text.empty()) {
        error = "CPU list is empty";
        return false;
    }

    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t comma = text.find(',', offset);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        const std::string_view token = text.substr(offset, end - offset);
        const std::size_t dash = token.find('-');

        unsigned first = 0;
        unsigned last = 0;
        if (dash == std::string_view::npos) {
            if (!ParseCpu(token, first)) {
                error = "invalid CPU list token: " + std::string(token);
                return false;
            }
            last = first;
        } else if (!ParseCpu(token.substr(0, dash), first) ||
                   !ParseCpu(token.substr(dash + 1), last) || first > last) {
            error = "invalid CPU range: " + std::string(token);
            return false;
        }

        if (last - first > 65535U) {
            error = "CPU range is too large: " + std::string(token);
            return false;
        }
        for (unsigned cpu = first;; ++cpu) {
            cpus.push_back(cpu);
            if (cpu == last) {
                break;
            }
        }

        if (comma == std::string_view::npos) {
            break;
        }
        offset = comma + 1;
        if (offset == text.size()) {
            error = "CPU list has an empty trailing token";
            return false;
        }
    }

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return true;
}

std::vector<unsigned> DiscoverCpuAffinityForPci(std::string_view pci_bus_id)
{
    const std::vector<unsigned> allowed = CurrentProcessCpus();
    if (!IsPciBusId(pci_bus_id)) {
        return allowed;
    }

    const std::string device_path =
        "/sys/bus/pci/devices/" + std::string(pci_bus_id);
    std::string cpu_list = ReadFirstLine(device_path + "/local_cpulist");
    if (cpu_list.empty()) {
        const std::string node_text = ReadFirstLine(device_path + "/numa_node");
        int node = -1;
        if (!node_text.empty()) {
            const auto parsed = std::from_chars(
                node_text.data(), node_text.data() + node_text.size(), node, 10);
            if (parsed.ec == std::errc{} && node >= 0) {
                cpu_list = ReadFirstLine("/sys/devices/system/node/node" +
                                         std::to_string(node) + "/cpulist");
            }
        }
    }

    std::vector<unsigned> preferred;
    std::string error;
    if (!ParseCpuList(cpu_list, preferred, error)) {
        return allowed;
    }
    if (allowed.empty()) {
        return preferred;
    }
    const std::vector<unsigned> intersection = IntersectCpus(preferred, allowed);
    return intersection.empty() ? allowed : intersection;
}

int SelectTopologyCpu(const std::vector<unsigned>& cpus,
                      unsigned device_rank,
                      bool consumer)
{
    if (cpus.empty()) {
        return -1;
    }
    const std::size_t offset = static_cast<std::size_t>(device_rank) * 2U +
                               (consumer ? 1U : 0U);
    return static_cast<int>(cpus[offset % cpus.size()]);
}

bool PinCurrentThreadToCpu(int cpu, std::string& error)
{
    error.clear();
    if (cpu < 0) {
        return true;
    }
#ifdef _WIN32
    if (cpu >= static_cast<int>(sizeof(DWORD_PTR) * 8U)) {
        error = "CPU index is outside the current Windows processor group";
        return false;
    }
    const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << cpu;
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        error = "SetThreadAffinityMask failed";
        return false;
    }
#else
    if (cpu >= CPU_SETSIZE) {
        error = "CPU index exceeds CPU_SETSIZE";
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int status = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (status != 0) {
        error = std::strerror(status);
        return false;
    }
#endif
    return true;
}

} // namespace rckangaroo
