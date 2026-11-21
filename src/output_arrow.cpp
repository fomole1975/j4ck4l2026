#include "output.hpp"
#include "run_parquet.hpp"

#include <arrow/io/file.h>
#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/schema.h>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace shopping {
namespace {

using parquet::ConvertedType;
using parquet::Repetition;
using parquet::Type;
using parquet::schema::GroupNode;
using parquet::schema::NodePtr;
using parquet::schema::PrimitiveNode;

NodePtr field(const char* name, Type::type physical, ConvertedType::type logical = ConvertedType::NONE,
              Repetition::type repetition = Repetition::REQUIRED) {
    return PrimitiveNode::Make(name, repetition, physical, logical);
}

std::shared_ptr<GroupNode> table_schema(bool user) {
    parquet::schema::NodeVector fields;
    if (user) {
        fields = {field("user_id", Type::BYTE_ARRAY, ConvertedType::UTF8),
                  field("day", Type::INT32, ConvertedType::DATE),
                  field("sessions", Type::INT32, ConvertedType::UINT_32),
                  field("events", Type::INT32, ConvertedType::UINT_32),
                  field("revenue_cents", Type::INT64),
                  field("top_category", Type::BYTE_ARRAY, ConvertedType::UTF8, Repetition::OPTIONAL)};
    } else {
        fields = {field("campaign_id", Type::BYTE_ARRAY, ConvertedType::UTF8),
                  field("channel", Type::BYTE_ARRAY, ConvertedType::UTF8),
                  field("day", Type::INT32, ConvertedType::DATE),
                  field("sessions", Type::INT32, ConvertedType::UINT_32),
                  field("purchasing_sessions", Type::INT32, ConvertedType::UINT_32),
                  field("revenue_cents", Type::INT64),
                  field("assisted_sessions", Type::INT32, ConvertedType::UINT_32)};
    }
    return std::static_pointer_cast<GroupNode>(GroupNode::Make("schema", Repetition::REQUIRED, fields));
}

// Each worker owns one file and its buffers. Typed column writes bypass the
// allocation and copying needed to first construct a second table of Arrow arrays.
class PartFile {
public:
    PartFile(const std::filesystem::path& path, bool users, parquet::Compression::type codec) {
        auto opened = arrow::io::FileOutputStream::Open(path.string());
        if (!opened.ok()) throw std::runtime_error(opened.status().ToString());
        stream = *opened;
        auto props = parquet::WriterProperties::Builder().disable_statistics()->disable_dictionary()->compression(codec)->build();
        writer = parquet::ParquetFileWriter::Open(stream, table_schema(users), std::move(props));
    }

    parquet::RowGroupWriter* next_group() { return writer->AppendRowGroup(); }

    void close() {
        writer->Close();
        auto status = stream->Close();
        if (!status.ok()) throw std::runtime_error(status.ToString());
    }

private:
    std::shared_ptr<arrow::io::FileOutputStream> stream;
    std::unique_ptr<parquet::ParquetFileWriter> writer;
};

template<class DataType, class Values>
void column(parquet::RowGroupWriter* group, const Values& values, std::size_t count, const int16_t* levels = nullptr) {
    auto* writer = static_cast<parquet::TypedColumnWriter<DataType>*>(group->NextColumn());
    writer->WriteBatch(static_cast<int64_t>(count), levels, nullptr, values.data());
}

struct ShortName { char bytes[9]; };

void write_users(const std::filesystem::path& path, const std::vector<UserSpan>& chunks, const Result& result,
                 parquet::Compression::type codec) {
    PartFile file(path, true, codec);
    std::size_t capacity = 0;
    for (const auto rows : chunks) capacity = std::min(parquet_group_rows, capacity + rows.size());
    std::vector<ShortName> names(capacity);
    std::vector<parquet::ByteArray> ids(capacity), categories(capacity);
    std::vector<std::int32_t> days(capacity), sessions(capacity), events(capacity);
    std::vector<std::int64_t> revenue(capacity);
    std::vector<std::int16_t> definitions(capacity);
    std::size_t filled = 0;
    std::size_t valid_categories = 0;
    auto flush = [&] {
        auto* group = file.next_group();
        column<parquet::ByteArrayType>(group, ids, filled);
        column<parquet::Int32Type>(group, days, filled);
        column<parquet::Int32Type>(group, sessions, filled);
        column<parquet::Int32Type>(group, events, filled);
        column<parquet::Int64Type>(group, revenue, filled);
        column<parquet::ByteArrayType>(group, categories, filled, definitions.data());
        group->Close();
        filled = valid_categories = 0;
    };
    // Engine shards are small and skewed. Fill whole row groups across shards
    // to amortize page headers, metadata, and compressor setup.
    for (auto rows : chunks) {
        while (!rows.empty()) {
            const auto n = std::min(capacity - filled, rows.size());
            for (const auto& r : rows.first(n)) {
                const auto i = filled++;
                ids[i] = r.user.view(names[i].bytes);
                days[i] = r.day;
                sessions[i] = std::bit_cast<int32_t>(r.sessions);
                events[i] = std::bit_cast<int32_t>(r.events);
                revenue[i] = r.revenue;
                definitions[i] = r.category >= 0;
                if (r.category >= 0) categories[valid_categories++] = result.category_names[r.category];
            }
            if (filled == capacity) flush();
            rows = rows.subspan(n);
        }
    }
    if (filled) flush();
    file.close();
}

void write_campaigns(const std::filesystem::path& path, const Result& result, parquet::Compression::type codec) {
    PartFile file(path, false, codec);
    std::span<const CampaignDaily> rows(result.campaigns);
    const auto capacity = std::min(parquet_group_rows, rows.size());
    std::vector<parquet::ByteArray> names(capacity), channels(capacity);
    std::vector<std::int32_t> days(capacity), sessions(capacity), purchasing(capacity), assists(capacity);
    std::vector<std::int64_t> revenue(capacity);
    while (!rows.empty()) {
        auto n = std::min(parquet_group_rows, rows.size());
        for (std::size_t i = 0; i < n; ++i) {
            const auto& r = rows[i];
            names[i] = result.campaign_names[r.campaign];
            channels[i] = result.channel_names[r.campaign];
            days[i] = r.day;
            sessions[i] = std::bit_cast<int32_t>(r.sessions);
            purchasing[i] = std::bit_cast<int32_t>(r.purchasing_sessions);
            revenue[i] = r.revenue;
            assists[i] = std::bit_cast<int32_t>(r.assists);
        }
        auto* group = file.next_group();
        column<parquet::ByteArrayType>(group, names, n);
        column<parquet::ByteArrayType>(group, channels, n);
        column<parquet::Int32Type>(group, days, n);
        column<parquet::Int32Type>(group, sessions, n);
        column<parquet::Int32Type>(group, purchasing, n);
        column<parquet::Int64Type>(group, revenue, n);
        column<parquet::Int32Type>(group, assists, n);
        group->Close();
        rows = rows.subspan(n);
    }
    file.close();
}

}

extern "C" void shopping_write_users(const std::filesystem::path& path, const std::vector<UserSpan>& spans,
                                      const Result& result, bool snappy) {
    write_users(path, spans, result, snappy ? parquet::Compression::SNAPPY : parquet::Compression::UNCOMPRESSED);
}
extern "C" void shopping_write_campaigns(const std::filesystem::path& path, const Result& result, bool snappy) {
    write_campaigns(path, result, snappy ? parquet::Compression::SNAPPY : parquet::Compression::UNCOMPRESSED);
}
}
