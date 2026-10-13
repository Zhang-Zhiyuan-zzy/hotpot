"""Advanced normalized dataset backend.

This namespace is intentionally separate from :mod:`hotpot.ds`.  It exposes
the relational record model and storage primitives used by adapters,
automation and forensic queries.  Most users should use the
compact facade in :mod:`hotpot.ds` instead.
"""

from ..database import DatasetSchemaError, DatasetStore, IngestResult
from ..levels import EvidenceLevel, QualityLevel
from ..models import (
    STANDARD_VERSION,
    CLAIM_RESOLUTION_TYPES,
    CLAIM_STATUSES,
    OBSERVATION_STATUSES,
    Asset,
    Claim,
    ClaimEvidence,
    ClaimMembership,
    ClaimResolution,
    ComputedDescriptor,
    Dataset as DatasetDefinition,
    DatasetBundle,
    DatasetRecord,
    DatasetRelease,
    DatasetStandardError,
    DatasetSubset,
    Entity,
    EntityRelation,
    EvidenceItem,
    ExternalIdentifier,
    Observation,
    ObservationEvidence,
    ObservationMembership,
    ObservationParticipant,
    PropertyType,
    RecordEntity,
    Source,
    Structure,
    StructureAsset,
    SubsetMembership,
    load_bundle,
    write_bundle,
)
from ..properties import PropertyRegistry, builtin_property_types
from ..public.certificate import Certificate, Configure
from ..public.store import Store
from ..quality import (
    QualityCertificate,
    QualityCheck,
    QualityGateError,
    assess_bundle,
    assess_file,
    load_certificate,
    verify_admission,
    write_certificate,
)

# A precise name avoids collision with the loaded public Dataset facade.
Dataset = DatasetDefinition

__all__ = [
    "STANDARD_VERSION",
    "CLAIM_RESOLUTION_TYPES",
    "CLAIM_STATUSES",
    "OBSERVATION_STATUSES",
    "Asset",
    "Claim",
    "ClaimEvidence",
    "ClaimMembership",
    "ClaimResolution",
    "ComputedDescriptor",
    "Certificate",
    "Configure",
    "Dataset",
    "DatasetBundle",
    "DatasetDefinition",
    "DatasetRecord",
    "DatasetRelease",
    "DatasetSchemaError",
    "DatasetStandardError",
    "DatasetStore",
    "DatasetSubset",
    "Entity",
    "EntityRelation",
    "EvidenceItem",
    "EvidenceLevel",
    "ExternalIdentifier",
    "IngestResult",
    "Observation",
    "ObservationEvidence",
    "ObservationMembership",
    "ObservationParticipant",
    "PropertyRegistry",
    "PropertyType",
    "QualityCertificate",
    "QualityCheck",
    "QualityGateError",
    "QualityLevel",
    "RecordEntity",
    "Source",
    "Structure",
    "StructureAsset",
    "Store",
    "SubsetMembership",
    "assess_bundle",
    "assess_file",
    "builtin_property_types",
    "load_bundle",
    "load_certificate",
    "verify_admission",
    "write_bundle",
    "write_certificate",
]
