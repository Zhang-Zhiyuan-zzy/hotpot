#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>


namespace hotpot::geometry {


using Point3 = std::array<double, 3>;
using IndexPair = std::array<std::size_t, 2>;


template <typename Value>
class ArrayView {
public:
    ArrayView() noexcept : data_(nullptr), size_(0) {}

    ArrayView(const Value* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    explicit ArrayView(const std::vector<Value>& values) noexcept
        : data_(values.data()), size_(values.size()) {}

    const Value* begin() const noexcept { return data_; }
    const Value* end() const noexcept {
        return size_ == 0 ? data_ : data_ + size_;
    }
    const Value& operator[](std::size_t index) const noexcept {
        return data_[index];
    }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

private:
    const Value* data_;
    std::size_t size_;
};


struct Line3 {
    Point3 origin;
    Point3 direction;
};


struct Segment3 {
    Point3 start;
    Point3 end;
};


struct Aabb {
    Point3 minimum;
    Point3 maximum;
};


enum class LineRelationKind : std::uint8_t {
    INTERSECTING,
    PARALLEL,
    COINCIDENT,
    SKEW,
    DEGENERATE,
    UNDETERMINED,
};


struct LineRelation {
    LineRelationKind kind;
    std::optional<double> distance;
    double parallel_measure;
};


struct PointSegmentMeasurement {
    double distance;
    Point3 closest_point;
    double parameter;
    bool segment_degenerate;
};


struct SegmentSegmentMeasurement {
    double distance;
    Point3 first_closest_point;
    Point3 second_closest_point;
    double first_parameter;
    double second_parameter;
    bool first_segment_degenerate;
    bool second_segment_degenerate;
};


struct PointPairDistance {
    std::size_t first_index;
    std::size_t second_index;
    double distance;
};


}  // namespace hotpot::geometry
