#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace rckangaroo {

bool ParseCpuList(std::string_view text,
                  std::vector<unsigned>& cpus,
                  std::string& error);

std::vector<unsigned> DiscoverCpuAffinityForPci(std::string_view pci_bus_id);

int SelectTopologyCpu(const std::vector<unsigned>& cpus,
                      unsigned device_rank,
                      bool consumer);

bool PinCurrentThreadToCpu(int cpu, std::string& error);

} // namespace rckangaroo
