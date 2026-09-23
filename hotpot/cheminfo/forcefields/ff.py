"""Force-field public façade for Python 3.10+ and current Open Babel."""

from . import utils as _utils


__all__ = _utils.__all__

# Public data contracts and exceptions have one defining module so that
# isinstance checks and pickle paths remain stable across both façades.
TrajectoryPath = _utils.TrajectoryPath
TrajectoryStart = _utils.TrajectoryStart
TrajectoryStage = _utils.TrajectoryStage
TrajectoryEvent = _utils.TrajectoryEvent
AtomIdentity = _utils.AtomIdentity
BondTopology = _utils.BondTopology
BondTopologyRevision = _utils.BondTopologyRevision
RingFrameEvidence = _utils.RingFrameEvidence
CoordinationFrameEvidence = _utils.CoordinationFrameEvidence
OptimizationFrameEvidence = _utils.OptimizationFrameEvidence
FrameEvidence = _utils.FrameEvidence
ForceFieldFrame = _utils.ForceFieldFrame
ForceFieldTrajectory = _utils.ForceFieldTrajectory
ForceFieldTrajectoryArchive = _utils.ForceFieldTrajectoryArchive
OptimizationAlgorithm = _utils.OptimizationAlgorithm
TerminationReason = _utils.TerminationReason
ForceFieldDiagnosticValue = _utils.ForceFieldDiagnosticValue
ForceFieldRunReport = _utils.ForceFieldRunReport
Build3DReport = _utils.Build3DReport
CandidateRejection = _utils.CandidateRejection
RingUntanglingReport = _utils.RingUntanglingReport
CoordinationBondRestorationReport = _utils.CoordinationBondRestorationReport
ComplexBuildDiagnostics = _utils.ComplexBuildDiagnostics
BuildWorkerResult = _utils.BuildWorkerResult
ForceFieldWorkflowReport = _utils.ForceFieldWorkflowReport
BuildAndOptimizeReport = _utils.BuildAndOptimizeReport
ComplexBuildReport = _utils.ComplexBuildReport
ForceFieldSetupReport = _utils.ForceFieldSetupReport
AcceptanceCheck = _utils.AcceptanceCheck
StructureAcceptanceThresholds = _utils.StructureAcceptanceThresholds
ForceFieldAcceptanceEvidence = _utils.ForceFieldAcceptanceEvidence
AtomTopologySignature = _utils.AtomTopologySignature
BondTopologySignature = _utils.BondTopologySignature
TopologyReference = _utils.TopologyReference
ForceFieldValidationReport = _utils.ForceFieldValidationReport
CoordinationEnvironment = _utils.CoordinationEnvironment
CoordinationGeometryCandidate = _utils.CoordinationGeometryCandidate
CoordinationGeometryResult = _utils.CoordinationGeometryResult
ForceFieldError = _utils.ForceFieldError
ForceFieldSetupError = _utils.ForceFieldSetupError
BuildWorkerError = _utils.BuildWorkerError
BuildTimeoutError = _utils.BuildTimeoutError
ComplexBuildError = _utils.ComplexBuildError
ComplexBuildWarning = _utils.ComplexBuildWarning
ComplexBuildWorkerError = _utils.ComplexBuildWorkerError
ComplexBuildTimeoutError = _utils.ComplexBuildTimeoutError
GeometryQualityError = _utils.GeometryQualityError
GeometryQualityWarning = _utils.GeometryQualityWarning

# These aliases are intentionally outside ``__all__`` but remain available for
# annotations and compatibility with the previous façade modules.
AcceptanceLevel = _utils.AcceptanceLevel
ForceFieldStage = _utils.ForceFieldStage

# Python 3.10+ uses the shared implementation and its current Open Babel worker
# targets directly.  Assignments retain the real signatures without wrappers.
capture_topology = _utils.capture_topology
evaluate_structure_acceptance = _utils.evaluate_structure_acceptance
is_structure_accepted = _utils.is_structure_accepted
perturb = _utils.perturb
collect_coordination_environments = _utils.collect_coordination_environments
prepare_coordination_geometry = _utils.prepare_coordination_geometry
build3d = _utils.build3d
optimize = _utils.optimize
build_complex3d = _utils.build_complex3d
optimize_complex = _utils.optimize_complex
complexes_build = _utils.complexes_build
build_and_optimize = _utils.build_and_optimize
auto_optimize = _utils.auto_optimize
