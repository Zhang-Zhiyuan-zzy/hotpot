#include "batch.hpp"
#include "cycle_surface.hpp"
#include "nonplanar_segment.hpp"
#include "nonplanar_surface.hpp"
#include "prepared_cycle.hpp"
#include "primitives.hpp"
#include "segment_cycle.hpp"
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
    if (
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(Value) != 0
    ) {
        throw py::value_error(std::string(name) + " must be memory-aligned");
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


std::vector<Point3> read_cycle(const py::array& array, const char* name) {
    std::vector<Point3> cycle = read_points(array, name);
    if (cycle.size() < detail::minimum_cycle_vertex_count) {
        throw py::value_error(
            std::string(name) + " requires at least three vertices"
        );
    }
    return cycle;
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


std::vector<std::size_t> read_indices(
    const py::array& array,
    const char* name
) {
    const std::int64_t* data = require_array<std::int64_t>(array, name, 1);
    std::vector<std::size_t> indices;
    indices.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t position = 0; position < array.shape(0); ++position) {
        if (data[position] < 0) {
            throw py::value_error(
                std::string(name) + " contains a negative index"
            );
        }
        indices.push_back(static_cast<std::size_t>(data[position]));
    }
    return indices;
}


std::vector<SegmentCyclePair> read_segment_cycle_pairs(
    const py::array& array,
    const char* name
) {
    const std::int64_t* data = require_array<std::int64_t>(array, name, 2);
    if (array.shape(1) != 2) {
        throw py::value_error(std::string(name) + " must have shape (N, 2)");
    }
    std::vector<SegmentCyclePair> pairs;
    pairs.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        const std::int64_t segment_index = data[row * 2];
        const std::int64_t cycle_index = data[row * 2 + 1];
        if (segment_index < 0 || cycle_index < 0) {
            throw py::value_error(
                std::string(name) + " contains a negative index"
            );
        }
        pairs.push_back({
            static_cast<std::size_t>(segment_index),
            static_cast<std::size_t>(cycle_index),
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


py::array_t<double> aabb_batch_array(const std::vector<Aabb>& bounds) {
    py::array_t<double> array({
        bounds.size(),
        std::size_t{2},
        std::size_t{3},
    });
    double* data = array.mutable_data();
    for (std::size_t index = 0; index < bounds.size(); ++index) {
        std::memcpy(data + index * 6, bounds[index].minimum.data(), 3 * sizeof(double));
        std::memcpy(
            data + index * 6 + 3,
            bounds[index].maximum.data(),
            3 * sizeof(double)
        );
    }
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


py::tuple point_tuple(const Point3& point) {
    return py::make_tuple(point[0], point[1], point[2]);
}


py::tuple point2_tuple(const Point2& point) {
    return py::make_tuple(point[0], point[1]);
}


py::list point_list(const std::vector<Point3>& points) {
    py::list result;
    for (const Point3& point : points) {
        result.append(point_tuple(point));
    }
    return result;
}


py::list point2_list(const std::vector<Point2>& points) {
    py::list result;
    for (const Point2& point : points) {
        result.append(point2_tuple(point));
    }
    return result;
}


py::object optional_point_tuple(const std::optional<Point3>& point) {
    if (point.has_value()) {
        return point_tuple(*point);
    }
    return py::none();
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


SurfaceEnumerationLimits make_surface_limits(
    std::size_t maximum_cycle_vertices,
    std::size_t maximum_surface_count,
    std::size_t maximum_segment_triangle_tests,
    std::size_t maximum_triangle_pair_tests
) {
    SurfaceEnumerationLimits limits{
        maximum_cycle_vertices,
        maximum_surface_count,
        maximum_segment_triangle_tests,
        maximum_triangle_pair_tests,
    };
    limits.validate();
    return limits;
}


}  // namespace


PYBIND11_MODULE(_geometry_native, module) {
    py::enum_<PlanarityKind>(module, "PlanarityKind")
        .value("PLANAR", PlanarityKind::PLANAR)
        .value("NONPLANAR", PlanarityKind::NONPLANAR)
        .value("DEGENERATE", PlanarityKind::DEGENERATE)
        .value("UNDETERMINED", PlanarityKind::UNDETERMINED);

    py::enum_<PolygonSimplicity>(module, "PolygonSimplicity")
        .value("SIMPLE", PolygonSimplicity::SIMPLE)
        .value("SELF_INTERSECTING", PolygonSimplicity::SELF_INTERSECTING)
        .value("UNDETERMINED", PolygonSimplicity::UNDETERMINED);

    py::enum_<PointCycleLocation>(module, "PointCycleLocation")
        .value("INTERIOR", PointCycleLocation::INTERIOR)
        .value("BOUNDARY", PointCycleLocation::BOUNDARY)
        .value("EXTERIOR", PointCycleLocation::EXTERIOR)
        .value("UNDETERMINED", PointCycleLocation::UNDETERMINED);

    py::enum_<SurfaceEmbeddingState>(module, "SurfaceEmbeddingState")
        .value("EMBEDDED", SurfaceEmbeddingState::EMBEDDED)
        .value(
            "PROVEN_NON_EMBEDDED",
            SurfaceEmbeddingState::PROVEN_NON_EMBEDDED
        )
        .value(
            "CONSTRUCTION_UNDETERMINED",
            SurfaceEmbeddingState::CONSTRUCTION_UNDETERMINED
        );

    py::enum_<NonplanarSurfaceCause>(module, "NonplanarSurfaceCause")
        .value(
            "INCOMPLETE_SURFACE_FAMILY",
            NonplanarSurfaceCause::INCOMPLETE_SURFACE_FAMILY
        )
        .value(
            "SURFACE_CONSTRUCTION",
            NonplanarSurfaceCause::SURFACE_CONSTRUCTION
        );

    py::enum_<PiercingState>(module, "PiercingState")
        .value("PIERCES", PiercingState::PIERCES)
        .value("DOES_NOT_PIERCE", PiercingState::DOES_NOT_PIERCE)
        .value("UNDETERMINED", PiercingState::UNDETERMINED);

    py::enum_<DetailLevel>(module, "DetailLevel")
        .value("STATE_ONLY", DetailLevel::STATE_ONLY)
        .value("ACTIONABLE", DetailLevel::ACTIONABLE)
        .value("FULL", DetailLevel::FULL);

    py::class_<SegmentCyclePair>(module, "SegmentCyclePair")
        .def(
            py::init<std::size_t, std::size_t>(),
            py::arg("segment_index"),
            py::arg("cycle_index")
        )
        .def_readonly("segment_index", &SegmentCyclePair::segment_index)
        .def_readonly("cycle_index", &SegmentCyclePair::cycle_index);

    py::enum_<SegmentCycleFeature>(module, "SegmentCycleFeature")
        .value("TRANSVERSE_INTERIOR", SegmentCycleFeature::TRANSVERSE_INTERIOR)
        .value(
            "LINE_EXTENSION_INTERIOR",
            SegmentCycleFeature::LINE_EXTENSION_INTERIOR
        )
        .value("CYCLE_EDGE_CONTACT", SegmentCycleFeature::CYCLE_EDGE_CONTACT)
        .value(
            "CYCLE_VERTEX_CONTACT",
            SegmentCycleFeature::CYCLE_VERTEX_CONTACT
        )
        .value(
            "SEGMENT_ENDPOINT_CONTACT",
            SegmentCycleFeature::SEGMENT_ENDPOINT_CONTACT
        )
        .value("COPLANAR_CONTACT", SegmentCycleFeature::COPLANAR_CONTACT);

    py::enum_<SegmentCycleIndeterminacy>(
        module,
        "SegmentCycleIndeterminacy"
    )
        .value(
            "NONFINITE_INPUT",
            SegmentCycleIndeterminacy::NONFINITE_INPUT
        )
        .value("NUMERIC_BAND", SegmentCycleIndeterminacy::NUMERIC_BAND)
        .value(
            "TOLERANCE_DOMAIN",
            SegmentCycleIndeterminacy::TOLERANCE_DOMAIN
        )
        .value(
            "DEGENERATE_CYCLE",
            SegmentCycleIndeterminacy::DEGENERATE_CYCLE
        )
        .value(
            "DEGENERATE_SEGMENT",
            SegmentCycleIndeterminacy::DEGENERATE_SEGMENT
        )
        .value(
            "DEGENERATE_TRIANGLE",
            SegmentCycleIndeterminacy::DEGENERATE_TRIANGLE
        )
        .value(
            "SELF_INTERSECTION",
            SegmentCycleIndeterminacy::SELF_INTERSECTION
        )
        .value(
            "SURFACE_DISAGREEMENT",
            SegmentCycleIndeterminacy::SURFACE_DISAGREEMENT
        )
        .value(
            "INCOMPLETE_SURFACE_FAMILY",
            SegmentCycleIndeterminacy::INCOMPLETE_SURFACE_FAMILY
        )
        .value(
            "SURFACE_CONSTRUCTION",
            SegmentCycleIndeterminacy::SURFACE_CONSTRUCTION
        );

    py::enum_<CycleSurfaceModel>(module, "CycleSurfaceModel")
        .value("PLANAR_POLYGON", CycleSurfaceModel::PLANAR_POLYGON)
        .value(
            "VERTEX_TRIANGULATION_FAMILY",
            CycleSurfaceModel::VERTEX_TRIANGULATION_FAMILY
        );

    py::class_<PlanarityMeasurement>(module, "PlanarityMeasurement")
        .def_readonly("kind", &PlanarityMeasurement::kind)
        .def_property_readonly(
            "centroid",
            [](const PlanarityMeasurement& result) {
                return point_tuple(result.centroid);
            }
        )
        .def_property_readonly(
            "normal",
            [](const PlanarityMeasurement& result) {
                return optional_point_tuple(result.normal);
            }
        )
        .def_property_readonly(
            "singular_values",
            [](const PlanarityMeasurement& result) {
                return point_tuple(result.singular_values);
            }
        )
        .def_readonly(
            "maximum_deviation",
            &PlanarityMeasurement::maximum_deviation
        )
        .def_readonly("rms_deviation", &PlanarityMeasurement::rms_deviation)
        .def_readonly("length_scale", &PlanarityMeasurement::length_scale)
        .def_readonly(
            "length_tolerance",
            &PlanarityMeasurement::length_tolerance
        );

    py::class_<PreparedPlanarCycle>(module, "PreparedPlanarCycle")
        .def_property_readonly(
            "coordinates",
            [](const PreparedPlanarCycle& cycle) {
                return point_list(cycle.coordinates());
            }
        )
        .def_property_readonly(
            "planarity",
            &PreparedPlanarCycle::planarity
        )
        .def_property_readonly(
            "tolerances",
            &PreparedPlanarCycle::tolerances
        )
        .def_property_readonly(
            "projection",
            [](const PreparedPlanarCycle& cycle) {
                return point2_list(cycle.projection());
            }
        )
        .def_property_readonly(
            "simplicity",
            &PreparedPlanarCycle::simplicity
        )
        .def_property_readonly(
            "has_planar_surface",
            &PreparedPlanarCycle::has_planar_surface
        )
        .def_property_readonly(
            "has_simple_planar_surface",
            &PreparedPlanarCycle::has_simple_planar_surface
        );

    py::class_<SurfaceEnumerationLimits>(module, "SurfaceEnumerationLimits")
        .def(
            py::init(&make_surface_limits),
            py::arg("maximum_cycle_vertices"),
            py::arg("maximum_surface_count"),
            py::arg("maximum_segment_triangle_tests"),
            py::arg("maximum_triangle_pair_tests")
        )
        .def_readonly(
            "maximum_cycle_vertices",
            &SurfaceEnumerationLimits::maximum_cycle_vertices
        )
        .def_readonly(
            "maximum_surface_count",
            &SurfaceEnumerationLimits::maximum_surface_count
        )
        .def_readonly(
            "maximum_segment_triangle_tests",
            &SurfaceEnumerationLimits::maximum_segment_triangle_tests
        )
        .def_readonly(
            "maximum_triangle_pair_tests",
            &SurfaceEnumerationLimits::maximum_triangle_pair_tests
        );

    py::class_<PreparedNonplanarSurfaceFamily>(
        module,
        "PreparedNonplanarSurfaceFamily"
    )
        .def_property_readonly(
            "coordinates",
            [](const PreparedNonplanarSurfaceFamily& family) {
                return point_list(family.coordinates());
            }
        )
        .def_property_readonly(
            "tolerances",
            &PreparedNonplanarSurfaceFamily::tolerances
        )
        .def_property_readonly(
            "limits",
            &PreparedNonplanarSurfaceFamily::limits
        )
        .def_property_readonly(
            "enumeration_complete",
            &PreparedNonplanarSurfaceFamily::enumeration_complete
        )
        .def_property_readonly(
            "enumerated_surface_count",
            &PreparedNonplanarSurfaceFamily::enumerated_surface_count
        )
        .def_property_readonly(
            "embedded_surface_count",
            &PreparedNonplanarSurfaceFamily::embedded_surface_count
        )
        .def_property_readonly(
            "proven_non_embedded_surface_count",
            &PreparedNonplanarSurfaceFamily::proven_non_embedded_surface_count
        )
        .def_property_readonly(
            "construction_undetermined_count",
            &PreparedNonplanarSurfaceFamily::construction_undetermined_count
        )
        .def_property_readonly(
            "triangle_pair_tests_used",
            &PreparedNonplanarSurfaceFamily::triangle_pair_tests_used
        )
        .def_property_readonly(
            "causes",
            [](const PreparedNonplanarSurfaceFamily& family) {
                py::set causes;
                for (const NonplanarSurfaceCause cause : family.causes()) {
                    causes.add(py::cast(cause));
                }
                return py::frozenset(causes);
            }
        )
        .def_property_readonly(
            "surface_states",
            &PreparedNonplanarSurfaceFamily::surface_states
        );

    py::class_<PreparedCycle>(module, "PreparedCycle")
        .def_property_readonly(
            "coordinates",
            [](const PreparedCycle& cycle) {
                return point_list(cycle.coordinates());
            }
        )
        .def_property_readonly(
            "bounds",
            [](const PreparedCycle& cycle) {
                return aabb_array(cycle.bounds());
            }
        )
        .def_property_readonly("planarity", &PreparedCycle::planarity)
        .def_property_readonly("tolerances", &PreparedCycle::tolerances)
        .def_property_readonly("limits", &PreparedCycle::limits)
        .def_property_readonly(
            "uses_nonplanar_surface_family",
            &PreparedCycle::uses_nonplanar_surface_family
        );

    py::class_<PreparedCycleBatch>(module, "PreparedCycleBatch")
        .def_property_readonly(
            "coordinate_count",
            &PreparedCycleBatch::coordinate_count
        )
        .def_property_readonly(
            "cycle_count",
            &PreparedCycleBatch::cycle_count
        )
        .def_property_readonly(
            "coordinates",
            [](const PreparedCycleBatch& cycles) {
                return point_list(cycles.coordinates());
            }
        )
        .def_property_readonly(
            "cycle_indices",
            &PreparedCycleBatch::cycle_indices
        )
        .def_property_readonly(
            "cycle_offsets",
            &PreparedCycleBatch::cycle_offsets
        )
        .def_property_readonly(
            "cycle_bounds",
            [](const PreparedCycleBatch& cycles) {
                return aabb_batch_array(cycles.cycle_bounds());
            }
        )
        .def(
            "cycle",
            &PreparedCycleBatch::cycle,
            py::arg("index"),
            py::return_value_policy::reference_internal
        );

    py::class_<ClosestCycleEdge>(module, "ClosestCycleEdge")
        .def_readonly("edge_index", &ClosestCycleEdge::edge_index)
        .def_readonly("distance", &ClosestCycleEdge::distance);

    py::class_<SurfaceFamilyEvidence>(module, "SurfaceFamilyEvidence")
        .def_readonly(
            "enumeration_complete",
            &SurfaceFamilyEvidence::enumeration_complete
        )
        .def_readonly(
            "enumerated_surface_count",
            &SurfaceFamilyEvidence::enumerated_surface_count
        )
        .def_readonly(
            "embedded_surface_count",
            &SurfaceFamilyEvidence::embedded_surface_count
        )
        .def_readonly(
            "proven_non_embedded_surface_count",
            &SurfaceFamilyEvidence::proven_non_embedded_surface_count
        )
        .def_readonly(
            "construction_undetermined_count",
            &SurfaceFamilyEvidence::construction_undetermined_count
        )
        .def_readonly(
            "intersecting_surface_count",
            &SurfaceFamilyEvidence::intersecting_surface_count
        )
        .def_readonly(
            "non_piercing_surface_count",
            &SurfaceFamilyEvidence::non_piercing_surface_count
        )
        .def_readonly(
            "evaluation_undetermined_count",
            &SurfaceFamilyEvidence::evaluation_undetermined_count
        )
        .def_readonly(
            "segment_triangle_tests_used",
            &SurfaceFamilyEvidence::segment_triangle_tests_used
        )
        .def_readonly(
            "triangle_pair_tests_used",
            &SurfaceFamilyEvidence::triangle_pair_tests_used
        );

    py::class_<SegmentCycleRelation>(module, "SegmentCycleRelation")
        .def_readonly("state", &SegmentCycleRelation::state)
        .def_readonly("features", &SegmentCycleRelation::features)
        .def_readonly(
            "indeterminacy_causes",
            &SegmentCycleRelation::indeterminacy_causes
        )
        .def_readonly("surface_model", &SegmentCycleRelation::surface_model)
        .def_property_readonly(
            "intersection_points",
            [](const SegmentCycleRelation& relation) {
                return point_list(relation.intersection_points);
            }
        )
        .def_readonly(
            "closest_boundary_edge",
            &SegmentCycleRelation::closest_boundary_edge
        )
        .def_readonly("surface_evidence", &SegmentCycleRelation::surface_evidence);

    py::class_<SegmentCycleScreening>(module, "SegmentCycleScreening")
        .def_readonly("state", &SegmentCycleScreening::state)
        .def_readonly("relation", &SegmentCycleScreening::relation)
        .def_readonly("aabb_separated", &SegmentCycleScreening::aabb_separated)
        .def_readonly("surface_complete", &SegmentCycleScreening::surface_complete);

    py::class_<SegmentCycleBatch>(module, "SegmentCycleBatch")
        .def_property_readonly("detail", &SegmentCycleBatch::detail)
        .def_property_readonly(
            "requested_pair_count",
            &SegmentCycleBatch::requested_pair_count
        )
        .def_property_readonly(
            "evaluated_pair_count",
            &SegmentCycleBatch::evaluated_pair_count
        )
        .def_property_readonly(
            "aabb_separated_pair_count",
            &SegmentCycleBatch::aabb_separated_pair_count
        )
        .def_property_readonly(
            "exact_pair_count",
            &SegmentCycleBatch::exact_pair_count
        )
        .def_property_readonly(
            "piercing_pair_count",
            &SegmentCycleBatch::piercing_pair_count
        )
        .def_property_readonly(
            "does_not_pierce_pair_count",
            &SegmentCycleBatch::does_not_pierce_pair_count
        )
        .def_property_readonly(
            "undetermined_pair_count",
            &SegmentCycleBatch::undetermined_pair_count
        )
        .def_property_readonly(
            "scan_complete",
            &SegmentCycleBatch::scan_complete
        )
        .def_property_readonly("states", &SegmentCycleBatch::states)
        .def_property_readonly(
            "aabb_separated",
            [](const SegmentCycleBatch& result) {
                return boolean_array(result.aabb_separated());
            }
        )
        .def_property_readonly(
            "surface_complete",
            [](const SegmentCycleBatch& result) {
                return boolean_array(result.surface_complete());
            }
        )
        .def_property_readonly(
            "relation_positions",
            &SegmentCycleBatch::relation_positions
        )
        .def_property_readonly("relations", &SegmentCycleBatch::relations);

    module.doc() = (
        "Open-Babel-independent C++ geometry kernels for Hotpot"
    );

    py::enum_<LineRelationKind>(module, "LineRelationKind")
        .value("INTERSECTING", LineRelationKind::INTERSECTING)
        .value("PARALLEL", LineRelationKind::PARALLEL)
        .value("COINCIDENT", LineRelationKind::COINCIDENT)
        .value("SKEW", LineRelationKind::SKEW)
        .value("DEGENERATE", LineRelationKind::DEGENERATE)
        .value("UNDETERMINED", LineRelationKind::UNDETERMINED);

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
        .def_property_readonly(
            "closest_point",
            [](const PointSegmentMeasurement& result) {
                return point_tuple(result.closest_point);
            }
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
        .def_property_readonly(
            "first_closest_point",
            [](const SegmentSegmentMeasurement& result) {
                return point_tuple(result.first_closest_point);
            }
        )
        .def_property_readonly(
            "second_closest_point",
            [](const SegmentSegmentMeasurement& result) {
                return point_tuple(result.second_closest_point);
            }
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
        "measure_planarity",
        [](const py::array& cycle, const NumericTolerances& tolerances) {
            const std::vector<Point3> coordinates = read_cycle(cycle, "cycle");
            py::gil_scoped_release release;
            return measure_planarity(ArrayView<Point3>(coordinates), tolerances);
        },
        py::arg("cycle"),
        py::arg("tolerances")
    );

    module.def(
        "prepare_planar_cycle",
        [](const py::array& cycle, const NumericTolerances& tolerances) {
            const std::vector<Point3> coordinates = read_cycle(cycle, "cycle");
            py::gil_scoped_release release;
            return prepare_planar_cycle(ArrayView<Point3>(coordinates), tolerances);
        },
        py::arg("cycle"),
        py::arg("tolerances")
    );

    module.def(
        "prepare_nonplanar_surface_family",
        [](const py::array& cycle,
           const NumericTolerances& tolerances,
           const SurfaceEnumerationLimits& limits) {
            const std::vector<Point3> coordinates = read_cycle(cycle, "cycle");
            py::gil_scoped_release release;
            return prepare_nonplanar_surface_family(
                ArrayView<Point3>(coordinates),
                tolerances,
                limits
            );
        },
        py::arg("cycle"),
        py::arg("tolerances"),
        py::arg("limits")
    );

    module.def(
        "prepare_cycle",
        [](const py::array& cycle,
           const NumericTolerances& tolerances,
           const SurfaceEnumerationLimits& limits) {
            const std::vector<Point3> coordinates = read_cycle(cycle, "cycle");
            py::gil_scoped_release release;
            return prepare_cycle(
                ArrayView<Point3>(coordinates),
                tolerances,
                limits
            );
        },
        py::arg("cycle"),
        py::arg("tolerances"),
        py::arg("limits")
    );

    module.def(
        "prepare_cycles",
        [](const py::array& coordinates,
           const py::array& cycle_indices,
           const py::array& cycle_offsets,
           const NumericTolerances& tolerances,
           const SurfaceEnumerationLimits& limits) {
            const std::vector<Point3> native_coordinates = read_points(
                coordinates,
                "coordinates"
            );
            const std::vector<std::size_t> native_indices = read_indices(
                cycle_indices,
                "cycle_indices"
            );
            const std::vector<std::size_t> native_offsets = read_indices(
                cycle_offsets,
                "cycle_offsets"
            );
            py::gil_scoped_release release;
            return prepare_cycles(
                ArrayView<Point3>(native_coordinates),
                ArrayView<std::size_t>(native_indices),
                ArrayView<std::size_t>(native_offsets),
                tolerances,
                limits
            );
        },
        py::arg("coordinates"),
        py::arg("cycle_indices"),
        py::arg("cycle_offsets"),
        py::arg("tolerances"),
        py::arg("limits")
    );

    module.def(
        "determine_segment_cycle_relations",
        [](const PreparedCycleBatch& cycles,
           const py::array& segments,
           const py::array& candidate_pairs) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            const std::vector<SegmentCyclePair> native_pairs =
                read_segment_cycle_pairs(candidate_pairs, "candidate_pairs");
            py::gil_scoped_release release;
            return determine_segment_cycle_relations(
                cycles,
                ArrayView<Segment3>(native_segments),
                ArrayView<SegmentCyclePair>(native_pairs)
            );
        },
        py::arg("cycles"),
        py::arg("segments"),
        py::arg("candidate_pairs")
    );

    module.def(
        "screen_segments",
        [](const PreparedCycleBatch& cycles,
           const py::array& segments,
           const py::array& candidate_pairs,
           DetailLevel detail,
           bool stop_after_confirmed) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            const std::vector<SegmentCyclePair> native_pairs =
                read_segment_cycle_pairs(candidate_pairs, "candidate_pairs");
            py::gil_scoped_release release;
            return screen_segments(
                cycles,
                ArrayView<Segment3>(native_segments),
                ArrayView<SegmentCyclePair>(native_pairs),
                detail,
                stop_after_confirmed
            );
        },
        py::arg("cycles"),
        py::arg("segments"),
        py::arg("candidate_pairs"),
        py::arg("detail"),
        py::arg("stop_after_confirmed") = false
    );

    module.def(
        "locate_point_in_planar_cycle",
        [](const py::array& point,
           const PreparedPlanarCycle& cycle,
           const py::array& plane_origin,
           const py::array& plane_normal) {
            const Point3 native_point = read_point(point, "point");
            const Point3 origin = read_point(plane_origin, "plane_origin");
            const Point3 normal = read_point(plane_normal, "plane_normal");
            py::gil_scoped_release release;
            return locate_point_in_planar_cycle(
                native_point,
                cycle,
                origin,
                normal
            );
        },
        py::arg("point"),
        py::arg("cycle"),
        py::arg("plane_origin"),
        py::arg("plane_normal")
    );

    module.def(
        "closest_cycle_edge",
        [](const PreparedPlanarCycle& cycle,
           const py::array& segment_start,
           const py::array& segment_end) {
            const Segment3 segment{
                read_point(segment_start, "segment_start"),
                read_point(segment_end, "segment_end"),
            };
            py::gil_scoped_release release;
            return closest_cycle_edge(
                cycle,
                segment
            );
        },
        py::arg("cycle"),
        py::arg("segment_start"),
        py::arg("segment_end")
    );

    module.def(
        "closest_cycle_edge",
        [](const PreparedCycle& cycle,
           const py::array& segment_start,
           const py::array& segment_end) {
            const Segment3 segment{
                read_point(segment_start, "segment_start"),
                read_point(segment_end, "segment_end"),
            };
            py::gil_scoped_release release;
            return closest_cycle_edge(cycle, segment);
        },
        py::arg("cycle"),
        py::arg("segment_start"),
        py::arg("segment_end")
    );

    module.def(
        "determine_planar_segment_cycle_relation",
        [](const py::array& segment_start,
           const py::array& segment_end,
           const PreparedPlanarCycle& cycle) {
            const Segment3 segment{
                read_point(segment_start, "segment_start"),
                read_point(segment_end, "segment_end"),
            };
            py::gil_scoped_release release;
            return determine_planar_segment_cycle_relation(
                segment,
                cycle
            );
        },
        py::arg("segment_start"),
        py::arg("segment_end"),
        py::arg("cycle")
    );

    module.def(
        "planar_segment_cycle_relations",
        [](const py::array& segments,
           const PreparedPlanarCycle& cycle) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            py::gil_scoped_release release;
            return planar_segment_cycle_relations(
                ArrayView<Segment3>(native_segments),
                cycle
            );
        },
        py::arg("segments"),
        py::arg("cycle")
    );

    module.def(
        "planar_segment_cycle_screenings",
        [](const py::array& segments,
           const PreparedPlanarCycle& cycle) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            py::gil_scoped_release release;
            return planar_segment_cycle_screenings(
                ArrayView<Segment3>(native_segments),
                cycle
            );
        },
        py::arg("segments"),
        py::arg("cycle")
    );

    module.def(
        "determine_nonplanar_segment_cycle_relation",
        [](const py::array& segment_start,
           const py::array& segment_end,
           const PreparedNonplanarSurfaceFamily& family) {
            const Segment3 segment{
                read_point(segment_start, "segment_start"),
                read_point(segment_end, "segment_end"),
            };
            py::gil_scoped_release release;
            return determine_nonplanar_segment_cycle_relation(
                segment,
                family
            );
        },
        py::arg("segment_start"),
        py::arg("segment_end"),
        py::arg("family")
    );

    module.def(
        "nonplanar_segment_cycle_relations",
        [](const py::array& segments,
           const PreparedNonplanarSurfaceFamily& family) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            py::gil_scoped_release release;
            return nonplanar_segment_cycle_relations(
                ArrayView<Segment3>(native_segments),
                family
            );
        },
        py::arg("segments"),
        py::arg("family")
    );

    module.def(
        "nonplanar_segment_cycle_screenings",
        [](const py::array& segments,
           const PreparedNonplanarSurfaceFamily& family) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            py::gil_scoped_release release;
            return nonplanar_segment_cycle_screenings(
                ArrayView<Segment3>(native_segments),
                family
            );
        },
        py::arg("segments"),
        py::arg("family")
    );

    module.def(
        "determine_segment_cycle_relation",
        [](const py::array& segment_start,
           const py::array& segment_end,
           const PreparedCycle& cycle) {
            const Segment3 segment{
                read_point(segment_start, "segment_start"),
                read_point(segment_end, "segment_end"),
            };
            py::gil_scoped_release release;
            return determine_segment_cycle_relation(segment, cycle);
        },
        py::arg("segment_start"),
        py::arg("segment_end"),
        py::arg("cycle")
    );

    module.def(
        "segment_cycle_relations",
        [](const py::array& segments, const PreparedCycle& cycle) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            py::gil_scoped_release release;
            return segment_cycle_relations(
                ArrayView<Segment3>(native_segments),
                cycle
            );
        },
        py::arg("segments"),
        py::arg("cycle")
    );

    module.def(
        "segment_cycle_screenings",
        [](const py::array& segments, const PreparedCycle& cycle) {
            const std::vector<Segment3> native_segments = read_segments(
                segments,
                "segments"
            );
            py::gil_scoped_release release;
            return segment_cycle_screenings(
                ArrayView<Segment3>(native_segments),
                cycle
            );
        },
        py::arg("segments"),
        py::arg("cycle")
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
