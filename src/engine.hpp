#pragma once
#include "key.hpp"
#include <cstdint>
#include <string_view>
#include <vector>

namespace shopping {
class InputFiles;

struct UserDaily {
    Key user;
    int32_t day;
    uint32_t sessions, events;
    int64_t revenue;
    int32_t category; // -1 when there was no purchase
};

struct CampaignDaily {
    uint32_t campaign;
    int32_t day;
    uint32_t sessions, purchasing_sessions, assists;
    int64_t revenue;
};

struct Result {
    std::vector<std::vector<UserDaily>> users;
    std::vector<CampaignDaily> campaigns;
    std::vector<std::string_view> category_names, campaign_names, channel_names;
};

Result run_engine(const InputFiles& input, unsigned workers);
}
