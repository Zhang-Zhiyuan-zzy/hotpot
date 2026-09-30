#include "placement_candidates.hpp"

#include "../../geometry/_native/construction.hpp"
#include "../../geometry/_native/vector_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>


namespace hotpot::forcefields::detail {
namespace {


using hotpot::geometry::Point3;
using hotpot::geometry::detail::add_scaled;
using hotpot::geometry::detail::finite;
using hotpot::geometry::detail::normalized;
using hotpot::geometry::detail::subtract;


constexpr double PI = 3.141592653589793238462643383279502884;


Point3 centroid(
    const std::vector<Coordinate>& coordinates,
    const std::vector<PlacementDonorTarget>& donors
) {
    Point3 center{0.0, 0.0, 0.0};
    for (const auto& donor : donors) {
        const auto& point = coordinates[static_cast<std::size_t>(
            donor.atom_index
        )];
        for (std::size_t axis = 0; axis < 3; ++axis) {
            center[axis] += point[axis];
        }
    }
    const double inverse = 1.0 / static_cast<double>(donors.size());
    for (double& value : center) {
        value *= inverse;
    }
    return center;
}


Point3 ligand_centroid(const PlacementEvaluationWorkspace& workspace) {
    Point3 center{0.0, 0.0, 0.0};
    std::size_t count = 0;
    for (std::size_t atom = 0; atom < workspace.input->atom_count(); ++atom) {
        if (std::find(
                workspace.input->metal_indices.begin(),
                workspace.input->metal_indices.end(),
                static_cast<std::int32_t>(atom)
            ) != workspace.input->metal_indices.end()) {
            continue;
        }
        const auto& point = (*workspace.coordinates)[atom];
        for (std::size_t axis = 0; axis < 3; ++axis) {
            center[axis] += point[axis];
        }
        ++count;
    }
    const double inverse = 1.0 / static_cast<double>(count);
    for (double& value : center) {
        value *= inverse;
    }
    return center;
}


std::vector<Point3> fibonacci_directions(std::size_t count) {
    std::vector<Point3> directions;
    directions.reserve(count);
    const double golden_angle = PI * (3.0 - std::sqrt(5.0));
    for (std::size_t index = 0; index < count; ++index) {
        const double z = 1.0 - 2.0 * (
            (static_cast<double>(index) + 0.5) / static_cast<double>(count)
        );
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double angle = static_cast<double>(index) * golden_angle;
        directions.push_back({
            radius * std::cos(angle),
            radius * std::sin(angle),
            z,
        });
    }
    return directions;
}


class ProposalCollector {
public:
    ProposalCollector(std::size_t maximum, double duplicate_tolerance)
        : maximum_(maximum), duplicate_tolerance_(duplicate_tolerance) {}

    void append(const Point3& point, PlacementProposalKind kind) {
        if (proposals_.size() >= maximum_ || !finite(point)) {
            return;
        }
        const auto key = std::array<std::int64_t, 3>{
            static_cast<std::int64_t>(std::llround(
                point[0] / duplicate_tolerance_
            )),
            static_cast<std::int64_t>(std::llround(
                point[1] / duplicate_tolerance_
            )),
            static_cast<std::int64_t>(std::llround(
                point[2] / duplicate_tolerance_
            )),
        };
        if (keys_.insert(key).second) {
            proposals_.push_back({point, kind, proposals_.size()});
        }
    }

    bool full() const noexcept {
        return proposals_.size() >= maximum_;
    }

    std::vector<PlacementProposal> finish() {
        return std::move(proposals_);
    }

private:
    std::size_t maximum_;
    double duplicate_tolerance_;
    std::set<std::array<std::int64_t, 3>> keys_;
    std::vector<PlacementProposal> proposals_;
};


void collect_sphere_inputs(
    const PlacementEvaluationWorkspace& workspace,
    std::vector<Point3>& centers,
    std::vector<double>& radii
) {
    centers.reserve(workspace.target.donors.size());
    radii.reserve(workspace.target.donors.size());
    for (const auto& donor : workspace.target.donors) {
        centers.push_back((*workspace.coordinates)[
            static_cast<std::size_t>(donor.atom_index)
        ]);
        radii.push_back(donor.target_distance_angstrom);
    }
}


Point3 refine_distance_fit(
    const Point3& point,
    const PlacementEvaluationWorkspace& workspace,
    const MetalPlacementOptions& options
) {
    std::vector<Point3> centers;
    std::vector<double> radii;
    collect_sphere_inputs(workspace, centers, radii);
    return hotpot::geometry::detail::fit_point_to_spheres(
        point,
        hotpot::geometry::ArrayView<Point3>(centers),
        hotpot::geometry::ArrayView<double>(radii),
        options.least_squares_iteration_count,
        options.geometry_tolerances
    );
}


Point3 linearized_least_squares_seed(
    const PlacementEvaluationWorkspace& workspace,
    const MetalPlacementOptions& options
) {
    std::vector<Point3> centers;
    std::vector<double> radii;
    collect_sphere_inputs(workspace, centers, radii);
    const Point3 fallback = centroid(
        *workspace.coordinates,
        workspace.target.donors
    );
    return hotpot::geometry::detail::linearized_sphere_fit_seed(
        hotpot::geometry::ArrayView<Point3>(centers),
        hotpot::geometry::ArrayView<double>(radii),
        fallback,
        options.geometry_tolerances
    );
}


void append_preferred_sphere_points(
    ProposalCollector& collector,
    const PlacementEvaluationWorkspace& workspace,
    const PlacementDonorTarget& donor,
    const Point3& donor_centroid,
    PlacementProposalKind kind
) {
    const Point3& donor_point = (*workspace.coordinates)[
        static_cast<std::size_t>(donor.atom_index)
    ];
    const Point3& input_metal = (*workspace.coordinates)[
        static_cast<std::size_t>(workspace.target.metal_index)
    ];
    const std::array<Point3, 8> raw_directions = {{
        subtract(donor_point, donor_centroid),
        subtract(input_metal, donor_point),
        {1.0, 0.0, 0.0},
        {-1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, -1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, -1.0},
    }};
    for (const Point3& raw : raw_directions) {
        const auto direction = normalized(raw);
        if (direction.has_value()) {
            collector.append(
                add_scaled(donor_point, *direction, donor.target_distance_angstrom),
                kind
            );
        }
    }
}


void append_sphere_intersection(
    ProposalCollector& collector,
    const PlacementEvaluationWorkspace& workspace,
    const MetalPlacementOptions& options
) {
    const auto& first = workspace.target.donors[0];
    const auto& second = workspace.target.donors[1];
    const Point3& first_point = (*workspace.coordinates)[
        static_cast<std::size_t>(first.atom_index)
    ];
    const Point3& second_point = (*workspace.coordinates)[
        static_cast<std::size_t>(second.atom_index)
    ];
    const auto circle = hotpot::geometry::detail::sphere_intersection_circle(
        first_point,
        first.target_distance_angstrom,
        second_point,
        second.target_distance_angstrom,
        options.geometry_tolerances
    );
    if (!circle.has_value()) {
        return;
    }
    for (
        std::size_t index = 0;
        index < options.sphere_intersection_count;
        ++index
    ) {
        const double angle = 2.0 * PI * static_cast<double>(index)
            / static_cast<double>(options.sphere_intersection_count);
        collector.append(
            hotpot::geometry::detail::point_on_circle(*circle, angle),
            PlacementProposalKind::SPHERE_INTERSECTION
        );
    }
}


}  // namespace


std::vector<PlacementProposal> generate_metal_placement_candidates(
    const PlacementEvaluationWorkspace& workspace,
    const MetalPlacementOptions& options
) {
    ProposalCollector collector(
        options.maximum_candidate_count,
        options.duplicate_tolerance_angstrom
    );
    const Point3 input_metal = (*workspace.coordinates)[
        static_cast<std::size_t>(workspace.target.metal_index)
    ];
    collector.append(input_metal, PlacementProposalKind::CURRENT);
    if (workspace.target.donors.empty() || collector.full()) {
        return collector.finish();
    }

    const Point3 donor_centroid = centroid(
        *workspace.coordinates,
        workspace.target.donors
    );
    if (workspace.target.donors.size() == 1) {
        append_preferred_sphere_points(
            collector,
            workspace,
            workspace.target.donors.front(),
            ligand_centroid(workspace),
            PlacementProposalKind::TARGET_SPHERE
        );
    } else if (workspace.target.donors.size() == 2) {
        append_sphere_intersection(collector, workspace, options);
    } else {
        collector.append(
            refine_distance_fit(
                linearized_least_squares_seed(workspace, options),
                workspace,
                options
            ),
            PlacementProposalKind::LEAST_SQUARES
        );
        collector.append(
            refine_distance_fit(donor_centroid, workspace, options),
            PlacementProposalKind::LEAST_SQUARES
        );
        for (const auto& donor : workspace.target.donors) {
            const Point3& point = (*workspace.coordinates)[
                static_cast<std::size_t>(donor.atom_index)
            ];
            const auto outward = normalized(subtract(point, donor_centroid));
            if (outward.has_value()) {
                collector.append(
                    refine_distance_fit(
                        add_scaled(
                            point,
                            *outward,
                            donor.target_distance_angstrom
                        ),
                        workspace,
                        options
                    ),
                    PlacementProposalKind::LEAST_SQUARES
                );
            }
        }
    }

    const auto directions = fibonacci_directions(
        options.fibonacci_direction_count
    );
    for (const Point3& direction : directions) {
        for (const auto& donor : workspace.target.donors) {
            const Point3& point = (*workspace.coordinates)[
                static_cast<std::size_t>(donor.atom_index)
            ];
            const Point3 proposal = add_scaled(
                point,
                direction,
                donor.target_distance_angstrom
            );
            collector.append(
                workspace.target.donors.size() >= 3
                    ? refine_distance_fit(proposal, workspace, options)
                    : proposal,
                PlacementProposalKind::FIBONACCI_FALLBACK
            );
            if (collector.full()) {
                return collector.finish();
            }
        }
    }
    return collector.finish();
}


}  // namespace hotpot::forcefields::detail
