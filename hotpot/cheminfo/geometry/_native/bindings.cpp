#include "primitives.hpp"
#include "spatial.hpp"
#include "tolerances.hpp"
#include "types.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace py = pybind11;
using namespace hotpot::geometry;


namespace {


template <typename Value>
const Value* require_array(
    const py::array& array,
    const char* name,
    int dimensions
) {
    if (!array.dtype().is(py::dtype::of<Value>())) {
        throw py::type_error(std::string(name) + " has an unexpected dtype");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (array.ndim() != dimensions) {
        throw py::value_error(
            std::string(name) + " has an unexpected number of dimensions"
        );
    }
    return static_cast<const Value*>(array.data());
}


Point3 read_point(const py::array& array, const char* name) {
    const double* data = require_array<double>(array, name, 1);
    if (array.shape(0) != 3) {
        throw py::value_error(std::string(name) + " must have shape (3,)");
    }
    return {data[0], data[1], data[2]};
}


std::vector<Point3> read_points(const py::array& array, const char* name) {
    const double* data = require_array<double>(array, name, 2);
    if (array.shape(1) != 3) {
        throw py::value_error(std::string(name) + " must have shape (N, 3)");
    }
    std::vector<Point3> points;
    points.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        points.push_back({
            data[row * 3],
            data[row * 3 + 1],
            data[row * 3 + 2],
        });
    }
    return points;
}


std::vector<Segment3> read_segments(
    const py::array& array,
    const char* name
) {
    const double* data = require_array<double>(array, name, 3);
    if (array.shape(1) != 2 || array.shape(2) != 3) {
        throw py::value_error(
            std::string(name) + " must have shape (N, 2, 3)"
        );
    }
    std::vector<Segment3> segments;
    segments.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * 6;
        segments.push_back({
            {data[offset], data[offset + 1], data[offset + 2]},
            {data[offset + 3], data[offset + 4], data[offset + 5]},
        });
    }
    return segments;
}


Aabb read_aabb(const py::array& array, const char* name) {
    const double* data = require_array<double>(array, name, 2);
    if (array.shape(0) != 2 || array.shape(1) != 3) {
        throw py::value_error(std::string(name) + " must have shape (2, 3)");
    }
    return {
        {data[0], data[1], data[2]},
        {data[3], data[4], data[5]},
    };
}


std::vector<Aabb> read_aabbs(const py::array& array, const char* name) {
    const double* data = require_array<double>(array, name, 3);
    if (array.shape(1) != 2 || array.shape(2) != 3) {
        throw py::value_error(
            std::string(name) + " must have shape (N, 2, 3)"
        );
    }
    std::vector<Aabb> bounds;
    bounds.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * 6;
        bounds.push_back({
            {data[offset], data[offset + 1], data[offset + 2]},
            {data[offset + 3], data[offset + 4], data[offset + 5]},
        });
    }
    return bounds;
}


std::vector<double> read_values(
    const py::array& array,
    const char* name
) {
    const double* data = require_array<double>(array, name, 1);
    return std::vector<double>(data, data + array.shape(0));
}


std::vector<IndexPair> read_index_pairs(
    const py::array& array,
    const char* name,
    std::size_t point_count
) {
    const std::int64_t* data = require_array<std::int64_t>(array, name, 2);
    if (array.shape(1) != 2) {
        throw py::value_error(std::string(name) + " must have shape (M, 2)");
    }
    std::vector<IndexPair> pairs;
    pairs.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        const std::int64_t first = data[row * 2];
        const std::int64_t second = data[row * 2 + 1];
        if (
            first < 0
            || second < 0
            || static_cast<std::size_t>(first) >= point_count
            || static_cast<std::size_t>(second) >= point_count
        ) {
            throw py::value_error(
                std::string(name) + " contains an out-of-range index"
            );
        }
        pairs.push_back({
            static_cast<std::size_t>(first),
            static_cast<std::size_t>(second),
        });
    }
    return pairs;
}


py::array_t<double> aabb_array(const Aabb& bounds) {
    py::array_t<double> array({std::size_t{2}, std::size_t{3}});
    std::memcpy(array.mutable_data(), bounds.minimum.data(), 3 * sizeof(double));
    std::memcpy(
        array.mutable_data() + 3,
        bounds.maximum.data(),
        3 * sizeof(double)
    );
    return array;
}


py::tuple point_pair_arrays(
    const std::vector<PointPairDistance>& measurements
) {
    py::array_t<std::int64_t> indices({
        measurements.size(),
        std::size_t{2},
    });
    py::array_t<double> distances(measurements.size());
    std::int64_t* index_data = indices.mutable_data();
    double* distance_data = distances.mutable_data();
    for (std::size_t row = 0; row < measurements.size(); ++row) {
        index_data[row * 2] = static_cast<std::int64_t>(
            measurements[row].first_index
        );
        index_data[row * 2 + 1] = static_cast<std::int64_t>(
            measurements[row].second_index
        );
        distance_data[row] = measurements[row].distance;
    }
    return py::make_tuple(std::move(indices), std::move(distances));
}


py::array_t<bool> boolean_array(const std::vector<std::uint8_t>& values) {
    py::array_t<bool> array(values.size());
    bool* data = array.mutable_data();
    for (std::size_t index = 0; index < values.size(); ++index) {
        data[index] = values[index] != 0;
    }
    return array;
}


py::array_t<std::int64_t> index_pair_array(
    const std::vector<IndexPair>& pairs
) {
    py::array_t<std::int64_t> array({pairs.size(), std::size_t{2}});
    std::int64_t* data = array.mutable_data();
    for (std::size_t row = 0; row < pairs.size(); ++row) {
        data[row * 2] = static_cast<std::int64_t>(pairs[row][0]);
        data[row * 2 + 1] = static_cast<std::int64_t>(pairs[row][1]);
    }
    return array;
}


NumericTolerances make_tolerances(
    double absolute_length,
    double relative_length,
    double parameter,
    double machine_epsilon_factor,
    double predicate_guard_factor,
    double planarity_factor,
    double winding_residual,
    double intersection_merge_factor,
    double aabb_padding_factor
) {
    NumericTolerances tolerances{
        absolute_length,
        relative_length,
        parameter,
        machine_epsilon_factor,
        predicate_guard_factor,
        planarity_factor,
        winding_residual,
        intersection_merge_factor,
        aabb_padding_factor,
    };
    tolerances.validate();
    return tolerances;
}


}  // namespace


PYBIND11_MODULE(_geometry_native, module) {
    module.doc() = (
        "Open-Babel-independent C++ geometry kernels for Hotpot"
    );

    py::enum_<LineRelationKind>(module, "LineRelationKind")
        .value("INTERSECTING", LineRelationKind::INTERSECTING)
        .value("PARALLEL", LineRelationKind::PARALLEL)
        .value("COINCIDENT", LineRelationKind::COINCIDENT)
        .value("SKEW", LineRelationKind::SKEW)
        .value("DEGENERATE", LineRelationKind::DEGENERATE)
        .value("UNDETERMINED", LineRelationKind::UNDETERMINED)
        .export_values();

    py::class_<NumericTolerances>(module, "NumericTolerances")
        .def(
            py::init(&make_tolerances),
            py::arg("absolute_length"),
            py::arg("relative_length"),
            py::arg("parameter"),
            py::arg("machine_epsilon_factor"),
            py::arg("predicate_guard_factor"),
            py::arg("planarity_factor"),
            py::arg("winding_residual"),
            py::arg("intersection_merge_factor"),
            py::arg("aabb_padding_factor")
        )
        .def_readonly("absolute_length", &NumericTolerances::absolute_length)
        .def_readonly("relative_length", &NumericTolerances::relative_length)
        .def_readonly("parameter", &NumericTolerances::parameter)
        .def_readonly(
            "machine_epsilon_factor",
            &NumericTolerances::machine_epsilon_factor
        )
        .def_readonly(
            "predicate_guard_factor",
            &NumericTolerances::predicate_guard_factor
        )
        .def_readonly("planarity_factor", &NumericTolerances::planarity_factor)
        .def_readonly("winding_residual", &NumericTolerances::winding_residual)
        .def_readonly(
            "intersection_merge_factor",
            &NumericTolerances::intersection_merge_factor
        )
        .def_readonly(
            "aabb_padding_factor",
            &NumericTolerances::aabb_padding_factor
        );

    py::class_<LineRelation>(module, "LineRelation")
        .def_readonly("kind", &LineRelation::kind)
        .def_readonly("distance", &LineRelation::distance)
        .def_readonly("parallel_measure", &LineRelation::parallel_measure);

    py::class_<PointSegmentMeasurement>(
        module,
        "PointSegmentMeasurement"
    )
        .def_readonly("distance", &PointSegmentMeasurement::distance)
        .def_readonly(
            "closest_point",
            &PointSegmentMeasurement::closest_point
        )
        .def_readonly("parameter", &PointSegmentMeasurement::parameter)
        .def_readonly(
            "segment_degenerate",
            &PointSegmentMeasurement::segment_degenerate
        );

    py::class_<SegmentSegmentMeasurement>(
        module,
        "SegmentSegmentMeasurement"
    )
        .def_readonly("distance", &SegmentSegmentMeasurement::distance)
        .def_readonly(
            "first_closest_point",
            &SegmentSegmentMeasurement::first_closest_point
        )
        .def_readonly(
            "second_closest_point",
            &SegmentSegmentMeasurement::second_closest_point
        )
        .def_readonly(
            "first_parameter",
            &SegmentSegmentMeasurement::first_parameter
        )
        .def_readonly(
            "second_parameter",
            &SegmentSegmentMeasurement::second_parameter
        )
        .def_readonly(
            "first_segment_degenerate",
            &SegmentSegmentMeasurement::first_segment_degenerate
        )
        .def_readonly(
            "second_segment_degenerate",
            &SegmentSegmentMeasurement::second_segment_degenerate
        );

    module.def(
        "determine_line_relation",
        [](const py::array& first_origin,
           const py::array& first_direction,
           const py::array& second_origin,
           const py::array& second_direction,
           const NumericTolerances& tolerances) {
            const Line3 first{
                read_point(first_origin, "first_origin"),
                read_point(first_direction, "first_direction"),
            };
            const Line3 second{
                read_point(second_origin, "second_origin"),
                read_point(second_direction, "second_direction"),
            };
            py::gil_scoped_release release;
            return determine_line_relation(first, second, tolerances);
        },
        py::arg("first_origin"),
        py::arg("first_direction"),
        py::arg("second_origin"),
        py::arg("second_direction"),
        py::arg("tolerances")
    );

    module.def(
        "line_distance",
        [](const py::array& first_origin,
           const py::array& first_direction,
           const py::array& second_origin,
           const py::array& second_direction,
           const NumericTolerances& tolerances) {
            const Line3 first{
                read_point(first_origin, "first_origin"),
                read_point(first_direction, "first_direction"),
            };
            const Line3 second{
                read_point(second_origin, "second_origin"),
                read_point(second_direction, "second_direction"),
            };
            py::gil_scoped_release release;
            return line_distance(first, second, tolerances);
        },
        py::arg("first_origin"),
        py::arg("first_direction"),
        py::arg("second_origin"),
        py::arg("second_direction"),
        py::arg("tolerances")
    );

    module.def(
        "point_segment_measurement",
        [](const py::array& point,
           const py::array& start,
           const py::array& end,
           const NumericTolerances& tolerances) {
            const Point3 native_point = read_point(point, "point");
            const Segment3 segment{
                read_point(start, "start"),
                read_point(end, "end"),
            };
            py::gil_scoped_release release;
            return point_segment_measurement(
                native_point,
                segment,
                tolerances
            );
        },
        py::arg("point"),
        py::arg("start"),
        py::arg("end"),
        py::arg("tolerances")
    );

    module.def(
        "point_segment_distance",
        [](const py::array& point,
           const py::array& start,
           const py::array& end,
           const NumericTolerances& tolerances) {
            const Point3 native_point = read_point(point, "point");
            const Segment3 segment{
                read_point(start, "start"),
                read_point(end, "end"),
            };
            py::gil_scoped_release release;
            return point_segment_distance(native_point, segment, tolerances);
        },
        py::arg("point"),
        py::arg("start"),
        py::arg("end"),
        py::arg("tolerances")
    );

    module.def(
        "segment_segment_measurement",
        [](const py::array& first_start,
           const py::array& first_end,
           const py::array& second_start,
           const py::array& second_end,
           const NumericTolerances& tolerances) {
            const Segment3 first{
                read_point(first_start, "first_start"),
                read_point(first_end, "first_end"),
            };
            const Segment3 second{
                read_point(second_start, "second_start"),
                read_point(second_end, "second_end"),
            };
            py::gil_scoped_release release;
            return segment_segment_measurement(first, second, tolerances);
        },
        py::arg("first_start"),
        py::arg("first_end"),
        py::arg("second_start"),
        py::arg("second_end"),
        py::arg("tolerances")
    );

    module.def(
        "segment_segment_distance",
        [](const py::array& first_start,
           const py::array& first_end,
           const py::array& second_start,
           const py::array& second_end,
           const NumericTolerances& tolerances) {
            const Segment3 first{
                read_point(first_start, "first_start"),
                read_point(first_end, "first_end"),
            };
            const Segment3 second{
                read_point(second_start, "second_start"),
                read_point(second_end, "second_end"),
            };
            py::gil_scoped_release release;
            return segment_segment_distance(first, second, tolerances);
        },
        py::arg("first_start"),
        py::arg("first_end"),
        py::arg("second_start"),
        py::arg("second_end"),
        py::arg("tolerances")
    );

    module.def(
        "point_pair_distances",
        [](const py::array& coordinates, const py::object& pair_indices) {
            const std::vector<Point3> points = read_points(
                coordinates,
                "coordinates"
            );
            std::vector<PointPairDistance> results;
            if (pair_indices.is_none()) {
                py::gil_scoped_release release;
                results = point_pair_distances(ArrayView<Point3>(points));
            } else {
                const std::vector<IndexPair> pairs = read_index_pairs(
                    py::cast<py::array>(pair_indices),
                    "pair_indices",
                    points.size()
                );
                py::gil_scoped_release release;
                results = point_pair_distances(
                    ArrayView<Point3>(points),
                    ArrayView<IndexPair>(pairs)
                );
            }
            return point_pair_arrays(results);
        },
        py::arg("coordinates"),
        py::arg("pair_indices") = py::none()
    );

    module.def(
        "find_point_pairs_below_distance",
        [](const py::array& coordinates, double threshold) {
            const std::vector<Point3> points = read_points(
                coordinates,
                "coordinates"
            );
            std::vector<PointPairDistance> results;
            {
                py::gil_scoped_release release;
                results = find_point_pairs_below_distance(
                    ArrayView<Point3>(points),
                    threshold
                );
            }
            return point_pair_arrays(results);
        },
        py::arg("coordinates"),
        py::arg("threshold")
    );

    module.def(
        "aabb_bounds",
        [](const py::array& coordinates) {
            const std::vector<Point3> points = read_points(
                coordinates,
                "coordinates"
            );
            Aabb bounds;
            {
                py::gil_scoped_release release;
                bounds = aabb_bounds(ArrayView<Point3>(points));
            }
            return aabb_array(bounds);
        },
        py::arg("coordinates")
    );

    module.def(
        "aabb_separation_mask",
        [](const py::array& first_bounds,
           const py::array& second_bounds,
           const py::array& paddings) {
            const std::vector<Aabb> first = read_aabbs(
                first_bounds,
                "first_bounds"
            );
            const std::vector<Aabb> second = read_aabbs(
                second_bounds,
                "second_bounds"
            );
            const std::vector<double> native_paddings = read_values(
                paddings,
                "paddings"
            );
            std::vector<std::uint8_t> separated;
            {
                py::gil_scoped_release release;
                separated = aabb_separation_mask(
                    ArrayView<Aabb>(first),
                    ArrayView<Aabb>(second),
                    ArrayView<double>(native_paddings)
                );
            }
            return boolean_array(separated);
        },
        py::arg("first_bounds"),
        py::arg("second_bounds"),
        py::arg("paddings")
    );

    module.def(
        "segment_aabb_separation_mask",
        [](const py::array& segments,
           const py::array& target_bounds,
           const py::array& paddings) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            const Aabb target = read_aabb(target_bounds, "target_bounds");
            const std::vector<double> native_paddings = read_values(
                paddings,
                "paddings"
            );
            std::vector<std::uint8_t> separated;
            {
                py::gil_scoped_release release;
                separated = segment_aabb_separation_mask(
                    ArrayView<Segment3>(native_segments),
                    target,
                    ArrayView<double>(native_paddings)
                );
            }
            return boolean_array(separated);
        },
        py::arg("segments"),
        py::arg("target_bounds"),
        py::arg("paddings")
    );

    module.def(
        "aabb_candidate_pairs",
        [](const py::array& first_bounds,
           const py::array& second_bounds,
           double padding) {
            const std::vector<Aabb> first = read_aabbs(
                first_bounds,
                "first_bounds"
            );
            const std::vector<Aabb> second = read_aabbs(
                second_bounds,
                "second_bounds"
            );
            std::vector<IndexPair> candidates;
            {
                py::gil_scoped_release release;
                candidates = aabb_candidate_pairs(
                    ArrayView<Aabb>(first),
                    ArrayView<Aabb>(second),
                    padding
                );
            }
            return index_pair_array(candidates);
        },
        py::arg("first_bounds"),
        py::arg("second_bounds"),
        py::arg("padding")
    );
}
