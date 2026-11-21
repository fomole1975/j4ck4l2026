#include "run_parquet.hpp"
#include "engine.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace shopping {
namespace {
using Bytes = std::string;

// A little-endian host can append Parquet numeric values directly. output.cpp
// selects Apache Parquet on other hosts. Large variable-width pages also use
// Apache's adaptive page subdivision, before our scratch buffers grow too large.
constexpr std::size_t max_string_page_bytes = 32 * 1024 * 1024;
struct PageLimit {};

enum PhysicalType : unsigned { int32_type = 1, int64_type = 2, byte_array = 6 };
enum ConvertedType : int { unannotated = -1, utf8 = 0, date = 6, uint32_type = 13 };
enum Encoding : unsigned { plain = 0, rle = 3, dictionary_indices = 8 };
enum PageType : unsigned { data_page = 0, dictionary_page = 2 };
constexpr unsigned uncompressed_codec = 0;

// Thrift compact protocol uses base-128 varints and field-ID deltas. Every
// metadata integer we write is non-negative, so ZigZag encoding is n << 1.

void varint(Bytes &out, uint64_t n) {
    while (n >= 128) {
        out.push_back(char((n & 127) | 128));
        n >>= 7;
    }
    out.push_back(char(n));
}
void text(Bytes &out, std::string_view s) {
    varint(out, s.size());
    out.append(s);
}

struct CompactProtocol {
    enum Type { I32 = 5, I64 = 6, Binary = 8, List = 9, Struct = 12 };
    Bytes &out;
    unsigned last = 0;
    void field(unsigned id, Type type) {
        // All schema fields below are emitted in ascending order with delta <= 15.
        out.push_back(char(((id - last) << 4) | type));
        last = id;
    }
    void integer(unsigned id, Type type, uint64_t n) {
        field(id, type);
        varint(out, n << 1);
    }
    void string(unsigned id, std::string_view s) {
        field(id, Binary);
        text(out, s);
    }
    void list(unsigned id, Type element, size_t count) {
        field(id, List);
        out.push_back(char((std::min<size_t>(15, count) << 4) | element));
        if (count >= 15)
            varint(out, count);
    }
    void close() {
        out.push_back(0);
    }
};

struct Column {
    const char *name;
    unsigned physical;
    int converted;
    bool optional = false;
};
constexpr std::array user_schema{Column{"user_id", byte_array, utf8},
                                 Column{"day", int32_type, date},
                                 Column{"sessions", int32_type, uint32_type},
                                 Column{"events", int32_type, uint32_type},
                                 Column{"revenue_cents", int64_type, unannotated},
                                 Column{"top_category", byte_array, utf8, true}};
constexpr std::array campaign_schema{Column{"campaign_id", byte_array, utf8},
                                     Column{"channel", byte_array, utf8},
                                     Column{"day", int32_type, date},
                                     Column{"sessions", int32_type, uint32_type},
                                     Column{"purchasing_sessions", int32_type, uint32_type},
                                     Column{"revenue_cents", int64_type, unannotated},
                                     Column{"assisted_sessions", int32_type, uint32_type}};

template <class T> void append(Bytes &out, T value) {
    out.append(reinterpret_cast<const char *>(&value), sizeof(value));
}
void append_string(Bytes &out, std::string_view s) {
    if (s.size() > max_string_page_bytes - sizeof(uint32_t) ||
        out.size() > max_string_page_bytes - sizeof(uint32_t) - s.size())
        throw PageLimit{};
    append(out, uint32_t(s.size()));
    out.append(s);
}

struct Dictionary {
    Bytes bytes;
    uint32_t entries = 0;
};

template <size_t N> class ParquetPart {
    struct Chunk {
        uint64_t offset, data_offset, size;
        bool dictionary;
    };
    struct Group {
        size_t rows;
        std::array<Chunk, N> columns;
    };
    const std::array<Column, N> &schema;
    std::filesystem::path path_;
    Bytes file_buf;
    uint64_t rows = 0;
    std::vector<Group> groups;
    void write(std::string_view bytes) {
        file_buf.append(bytes);
    }

  public:
    ParquetPart(const std::filesystem::path &path, const std::array<Column, N> &schema)
        : schema(schema), path_(path) {
        file_buf.reserve(4 * 1024 * 1024);
        write("PAR1");
    }
    void group(size_t n, const std::array<Bytes, N> &values, std::string_view definition_levels = {},
               const std::array<Dictionary, N> *dictionaries = nullptr) {
        Group group_info{n, {}};
        for (size_t i = 0; i < N; ++i) {
            const uint64_t chunk_begin = file_buf.size();
            const auto *dict = dictionaries ? &(*dictionaries)[i] : nullptr;
            bool has_dictionary = dict && dict->entries;
            if (has_dictionary) {
                Bytes header;
                CompactProtocol page{header};
                page.integer(1, CompactProtocol::I32, dictionary_page);
                page.integer(2, CompactProtocol::I32, dict->bytes.size());
                page.integer(3, CompactProtocol::I32, dict->bytes.size());
                page.field(7, CompactProtocol::Struct);
                CompactProtocol dictionary_header{header};
                dictionary_header.integer(1, CompactProtocol::I32, dict->entries);
                dictionary_header.integer(2, CompactProtocol::I32, plain);
                dictionary_header.close();
                page.close();
                write(header);
                write(dict->bytes);
            }
            const std::string_view levels = schema[i].optional ? definition_levels : std::string_view{};
            const size_t payload = values[i].size() + levels.size();
            if (payload > INT32_MAX)
                throw PageLimit{};
            Bytes buffer;
            CompactProtocol page{buffer};
            page.integer(1, CompactProtocol::I32, data_page);
            page.integer(2, CompactProtocol::I32, payload);
            page.integer(3, CompactProtocol::I32, payload);
            page.field(5, CompactProtocol::Struct);
            CompactProtocol data{buffer};
            data.integer(1, CompactProtocol::I32, n);
            data.integer(2, CompactProtocol::I32, has_dictionary ? dictionary_indices : plain);
            data.integer(3, CompactProtocol::I32, rle);
            data.integer(4, CompactProtocol::I32, rle);
            data.close();
            page.close();
            const uint64_t data_off = file_buf.size();
            group_info.columns[i] = {chunk_begin, data_off, data_off - chunk_begin + buffer.size() + payload,
                                     has_dictionary};
            write(buffer);
            if (!levels.empty())
                write(levels);
            write(values[i]);
        }
        groups.push_back(group_info);
        rows += n;
    }
    void close() {
        Bytes buffer;
        CompactProtocol meta{buffer};
        meta.integer(1, CompactProtocol::I32, 1);
        meta.list(2, CompactProtocol::Struct, N + 1);
        CompactProtocol root{buffer};
        root.string(4, "schema");
        root.integer(5, CompactProtocol::I32, N);
        root.close();
        for (auto c : schema) {
            CompactProtocol leaf{buffer};
            leaf.integer(1, CompactProtocol::I32, c.physical);
            leaf.integer(3, CompactProtocol::I32, c.optional);
            leaf.string(4, c.name);
            if (c.converted >= 0)
                leaf.integer(6, CompactProtocol::I32, c.converted);
            leaf.close();
        }
        meta.integer(3, CompactProtocol::I64, rows);
        meta.list(4, CompactProtocol::Struct, groups.size());
        for (const auto &group_info : groups) {
            CompactProtocol row_group{buffer};
            row_group.list(1, CompactProtocol::Struct, N);
            uint64_t bytes = 0;
            for (size_t i = 0; i < N; ++i) {
                auto col = schema[i];
                auto column_chunk = group_info.columns[i];
                CompactProtocol chunk{buffer};
                chunk.integer(2, CompactProtocol::I64, 0);
                chunk.field(3, CompactProtocol::Struct);
                CompactProtocol metadata{buffer};
                metadata.integer(1, CompactProtocol::I32, col.physical);
                metadata.list(2, CompactProtocol::I32,
                              1 + unsigned(col.optional) + unsigned(column_chunk.dictionary));
                varint(buffer, plain << 1);
                if (column_chunk.dictionary)
                    varint(buffer, dictionary_indices << 1);
                if (col.optional)
                    varint(buffer, rle << 1);
                metadata.list(3, CompactProtocol::Binary, 1);
                text(buffer, col.name);
                metadata.integer(4, CompactProtocol::I32, uncompressed_codec);
                metadata.integer(5, CompactProtocol::I64, group_info.rows);
                metadata.integer(6, CompactProtocol::I64, column_chunk.size);
                metadata.integer(7, CompactProtocol::I64, column_chunk.size);
                metadata.integer(9, CompactProtocol::I64, column_chunk.data_offset);
                if (column_chunk.dictionary)
                    metadata.integer(11, CompactProtocol::I64, column_chunk.offset);
                metadata.close();
                chunk.close();
                bytes += column_chunk.size;
            }
            row_group.integer(2, CompactProtocol::I64, bytes);
            row_group.integer(3, CompactProtocol::I64, group_info.rows);
            row_group.close();
        }
        meta.close();
        if (buffer.size() > UINT32_MAX)
            throw PageLimit{};
        const uint32_t footer = buffer.size();
        append(buffer, footer);
        buffer.append("PAR1");
        write(buffer);

        int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0)
            throw std::system_error(errno, std::generic_category(), path_.string());
        const char *ptr = file_buf.data();
        size_t rem = file_buf.size();
        while (rem > 0) {
            ssize_t written = ::write(fd, ptr, rem);
            if (written < 0) {
                if (errno == EINTR) continue;
                int ec = errno;
                ::close(fd);
                throw std::system_error(ec, std::generic_category(), path_.string());
            }
            ptr += written;
            rem -= static_cast<size_t>(written);
        }
        if (::close(fd) != 0)
            throw std::system_error(errno, std::generic_category(), path_.string());
    }
};

void write_user_file(const std::filesystem::path &p, const std::vector<UserSpan> &spans,
                     const Result &result) {
    ParquetPart file(p, user_schema);
    std::array<Bytes, 6> columns;
    for (auto &s : columns)
        s.reserve(parquet_group_rows * 8);
    std::array<uint8_t, parquet_group_rows / 8> bits{};
    std::array<Dictionary, 6> dictionaries{};
    std::vector<uint32_t> runs, days;
    runs.reserve(parquet_group_rows);
    days.reserve(parquet_group_rows);
    Key previous;
    int32_t first_day = INT32_MAX, last_day = INT32_MIN;
    size_t n = 0;
    auto flush = [&] {
        if (!n)
            return;
        // Optional top_category has one bit per row. The V1 definition-level
        // stream starts with a four-byte length and one bit-packed run header.
        const size_t bytes = (n + 7) / 8;
        Bytes levels;
        levels.resize(4);
        varint(levels, bytes * 2 + 1);
        levels.append(reinterpret_cast<const char *>(bits.data()), bytes);
        uint32_t len = levels.size() - 4;
        std::memcpy(levels.data(), &len, 4);
        // Consecutive user IDs share one dictionary value. Parquet RLE stores
        // the repeat count as an even varint followed by the fixed-width index.
        auto &user_dictionary = dictionaries[0];
        user_dictionary.entries = runs.size();
        unsigned width = std::max<unsigned>(1, std::bit_width(user_dictionary.entries - 1));
        columns[0].push_back(char(width));
        for (uint32_t j = 0; j < runs.size(); ++j) {
            varint(columns[0], uint64_t(runs[j]) * 2);
            const unsigned index_bytes = (width + 7) / 8;
            columns[0].append(reinterpret_cast<const char *>(&j), index_bytes);
        }
        const int64_t range = int64_t(last_day) - first_day;
        // A narrow date range needs at most eight bits per dictionary index.
        // Hybrid bit-packed runs contain complete groups of eight indices, LSB first.
        if (range <= 255) {
            auto &dates = dictionaries[1];
            dates.entries = uint32_t(range) + 1;
            for (int64_t day = first_day; day <= last_day; ++day)
                append(dates.bytes, int32_t(day));
            auto &encoded = columns[1];
            unsigned width = std::max<unsigned>(1, std::bit_width(uint32_t(range)));
            encoded.push_back(char(width));
            const unsigned blocks = (n + 7) / 8;
            varint(encoded, uint64_t(blocks) * 2 + 1);
            uint64_t accumulator = 0;
            unsigned occupied = 0;
            for (size_t j = 0; j < blocks * 8; ++j) {
                uint32_t index = j < n ? days[j] - uint32_t(first_day) : 0;
                accumulator |= uint64_t(index) << occupied;
                occupied += width;
                if (occupied >= 8) {
                    encoded.push_back(char(accumulator));
                    accumulator >>= 8;
                    occupied -= 8;
                }
            }
        } else
            columns[1].append(reinterpret_cast<const char *>(days.data()),
                              days.size() * sizeof(days[0]));
        file.group(n, columns, levels, &dictionaries);
        for (auto &d : dictionaries) {
            d.bytes.clear();
            d.entries = 0;
        }
        days.clear();
        runs.clear();
        first_day = INT32_MAX;
        last_day = INT32_MIN;
        for (auto &s : columns)
            s.clear();
        std::memset(bits.data(), 0, bytes);
        n = 0;
    };
    for (auto span : spans)
        for (auto &r : span) {
            if (runs.empty() || r.user != previous) {
                char storage[9];
                append_string(dictionaries[0].bytes, r.user.view(storage));
                runs.push_back(1);
                previous = r.user;
            } else
                ++runs.back();
            first_day = std::min(first_day, r.day);
            last_day = std::max(last_day, r.day);
            days.push_back(uint32_t(r.day));
            append(columns[2], r.sessions);
            append(columns[3], r.events);
            append(columns[4], r.revenue);
            if (r.category >= 0) {
                bits[n / 8] |= uint8_t(1u << (n % 8));
                append_string(columns[5], result.category_names[r.category]);
            }
            if (++n == parquet_group_rows)
                flush();
        }
    flush();
    file.close();
}

void write_campaign_file(const std::filesystem::path &p, const Result &result) {
    ParquetPart file(p, campaign_schema);
    std::array<Bytes, 7> columns;
    size_t n = 0;
    for (const auto &r : result.campaigns) {
        append_string(columns[0], result.campaign_names[r.campaign]);
        append_string(columns[1], result.channel_names[r.campaign]);
        append(columns[2], r.day);
        append(columns[3], r.sessions);
        append(columns[4], r.purchasing_sessions);
        append(columns[5], r.revenue);
        append(columns[6], r.assists);
        if (++n == parquet_group_rows) {
            file.group(n, columns);
            for (auto &s : columns)
                s.clear();
            n = 0;
        }
    }
    if (n)
        file.group(n, columns);
    file.close();
}
} // namespace

bool try_run_encoded_users(const std::filesystem::path &path, const std::vector<UserSpan> &spans,
                           const Result &result) {
    try {
        write_user_file(path, spans, result);
        return true;
    } catch (const PageLimit &) {
        return false;
    }
}

bool try_plain_campaigns(const std::filesystem::path &path, const Result &result) {
    try {
        write_campaign_file(path, result);
        return true;
    } catch (const PageLimit &) {
        return false;
    }
}
} // namespace shopping
