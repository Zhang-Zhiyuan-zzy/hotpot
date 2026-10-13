PRAGMA foreign_keys = ON;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE entries (
    refcode TEXT PRIMARY KEY,
    crystal_grade TEXT NOT NULL,
    formula_text TEXT,
    has_oxidation_text INTEGER NOT NULL CHECK (has_oxidation_text IN (0, 1)),
    has_spin_text INTEGER NOT NULL CHECK (has_spin_text IN (0, 1))
) WITHOUT ROWID;

CREATE TABLE entry_cores (
    refcode TEXT NOT NULL REFERENCES entries(refcode) ON DELETE CASCADE,
    core_id TEXT NOT NULL,
    PRIMARY KEY (refcode, core_id)
) WITHOUT ROWID;

CREATE TABLE sites (
    site_id TEXT PRIMARY KEY,
    refcode TEXT NOT NULL REFERENCES entries(refcode) ON DELETE CASCADE,
    core_id TEXT NOT NULL,
    representative_label TEXT NOT NULL,
    element TEXT NOT NULL,
    coordination_number INTEGER NOT NULL CHECK (coordination_number >= 0),
    primary_core_class TEXT NOT NULL,
    core_classification_status TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX sites_refcode_idx ON sites(refcode);
CREATE INDEX sites_element_cn_idx ON sites(element, coordination_number);
CREATE INDEX sites_core_idx ON sites(core_id);

CREATE TABLE entry_oxidation_evidence (
    refcode TEXT NOT NULL REFERENCES entries(refcode) ON DELETE CASCADE,
    element TEXT NOT NULL,
    grade TEXT NOT NULL,
    PRIMARY KEY (refcode, element)
) WITHOUT ROWID;

CREATE INDEX entry_oxidation_grade_idx ON entry_oxidation_evidence(grade);

CREATE TABLE entry_oxidation_states (
    refcode TEXT NOT NULL,
    element TEXT NOT NULL,
    oxidation_state INTEGER NOT NULL,
    PRIMARY KEY (refcode, element, oxidation_state),
    FOREIGN KEY (refcode, element)
        REFERENCES entry_oxidation_evidence(refcode, element) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE TABLE oxidation_mentions (
    mention_id INTEGER PRIMARY KEY,
    refcode TEXT NOT NULL REFERENCES entries(refcode) ON DELETE CASCADE,
    element TEXT NOT NULL,
    source_tag TEXT NOT NULL,
    source_text TEXT NOT NULL,
    matched_text TEXT NOT NULL
);

CREATE INDEX oxidation_mentions_refcode_idx ON oxidation_mentions(refcode);
CREATE INDEX oxidation_mentions_element_idx ON oxidation_mentions(element);

CREATE TABLE oxidation_mention_states (
    mention_id INTEGER NOT NULL REFERENCES oxidation_mentions(mention_id) ON DELETE CASCADE,
    oxidation_state INTEGER NOT NULL,
    PRIMARY KEY (mention_id, oxidation_state)
) WITHOUT ROWID;

CREATE TABLE site_oxidation_evidence (
    site_id TEXT PRIMARY KEY REFERENCES sites(site_id) ON DELETE CASCADE,
    grade TEXT NOT NULL,
    evidence_count INTEGER NOT NULL CHECK (evidence_count > 0)
) WITHOUT ROWID;

CREATE INDEX site_oxidation_grade_idx ON site_oxidation_evidence(grade);

CREATE TABLE site_oxidation_states (
    site_id TEXT NOT NULL REFERENCES site_oxidation_evidence(site_id) ON DELETE CASCADE,
    oxidation_state INTEGER NOT NULL,
    PRIMARY KEY (site_id, oxidation_state)
) WITHOUT ROWID;

CREATE TABLE spin_mentions (
    mention_id INTEGER PRIMARY KEY,
    refcode TEXT NOT NULL REFERENCES entries(refcode) ON DELETE CASCADE,
    source_tag TEXT NOT NULL,
    source_text TEXT NOT NULL,
    matched_text TEXT NOT NULL,
    spin_label TEXT NOT NULL
);

CREATE INDEX spin_mentions_refcode_idx ON spin_mentions(refcode);
CREATE INDEX spin_mentions_label_idx ON spin_mentions(spin_label);

CREATE TABLE entry_spin_evidence (
    refcode TEXT PRIMARY KEY REFERENCES entries(refcode) ON DELETE CASCADE,
    grade TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX entry_spin_grade_idx ON entry_spin_evidence(grade);

CREATE TABLE entry_spin_candidate_sites (
    refcode TEXT NOT NULL REFERENCES entry_spin_evidence(refcode) ON DELETE CASCADE,
    site_id TEXT NOT NULL REFERENCES sites(site_id) ON DELETE CASCADE,
    PRIMARY KEY (refcode, site_id)
) WITHOUT ROWID;

CREATE TABLE site_spin_evidence (
    site_id TEXT PRIMARY KEY REFERENCES sites(site_id) ON DELETE CASCADE,
    grade TEXT NOT NULL,
    evidence_count INTEGER NOT NULL CHECK (evidence_count > 0)
) WITHOUT ROWID;

CREATE INDEX site_spin_grade_idx ON site_spin_evidence(grade);

CREATE TABLE site_spin_labels (
    site_id TEXT NOT NULL REFERENCES site_spin_evidence(site_id) ON DELETE CASCADE,
    spin_label TEXT NOT NULL,
    PRIMARY KEY (site_id, spin_label)
) WITHOUT ROWID;

CREATE TABLE entry_spin_context_flags (
    refcode TEXT NOT NULL REFERENCES entry_spin_evidence(refcode) ON DELETE CASCADE,
    flag TEXT NOT NULL,
    source_text TEXT NOT NULL,
    matched_text TEXT NOT NULL,
    PRIMARY KEY (refcode, flag, source_text, matched_text)
) WITHOUT ROWID;

CREATE INDEX entry_spin_context_flag_idx ON entry_spin_context_flags(flag);

CREATE VIEW v_ox_t1_sites AS
SELECT s.refcode, s.core_id, s.site_id, s.representative_label, s.element,
       s.coordination_number, os.oxidation_state, s.primary_core_class,
       s.core_classification_status
FROM sites AS s
JOIN site_oxidation_evidence AS oe USING (site_id)
JOIN site_oxidation_states AS os USING (site_id)
WHERE oe.grade = 'OX-T1_EXPLICIT_SINGLE_SITE_CANDIDATE';

CREATE VIEW v_sp_t1_sites AS
SELECT s.refcode, s.core_id, s.site_id, s.representative_label, s.element,
       s.coordination_number, sl.spin_label, s.primary_core_class,
       s.core_classification_status
FROM sites AS s
JOIN site_spin_evidence AS se USING (site_id)
JOIN site_spin_labels AS sl USING (site_id)
WHERE se.grade = 'SP-T1_FIXED_SINGLE_METAL_CANDIDATE';

CREATE VIEW v_ox_t1_active_sites AS
SELECT * FROM v_ox_t1_sites WHERE core_classification_status = 'active';

CREATE VIEW v_sp_t1_active_sites AS
SELECT * FROM v_sp_t1_sites WHERE core_classification_status = 'active';

CREATE VIEW v_sp_t1_active_unqualified_sites AS
SELECT p.* FROM v_sp_t1_active_sites AS p
WHERE NOT EXISTS (
  SELECT 1 FROM entry_spin_context_flags AS f WHERE f.refcode = p.refcode
);

CREATE VIEW v_joint_t1_sites AS
SELECT o.refcode, o.core_id, o.site_id, o.representative_label, o.element,
       o.coordination_number, o.oxidation_state, p.spin_label,
       o.primary_core_class, o.core_classification_status
FROM v_ox_t1_sites AS o
JOIN v_sp_t1_sites AS p USING (site_id);

CREATE VIEW v_ox_t1_by_element AS
SELECT element, COUNT(*) AS site_count, COUNT(DISTINCT refcode) AS entry_count
FROM v_ox_t1_sites GROUP BY element;

CREATE VIEW v_sp_t1_by_element AS
SELECT element, spin_label, COUNT(*) AS site_count,
       COUNT(DISTINCT refcode) AS entry_count
FROM v_sp_t1_sites GROUP BY element, spin_label;
