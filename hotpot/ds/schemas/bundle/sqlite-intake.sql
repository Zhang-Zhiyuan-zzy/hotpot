CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    public_index INTEGER NOT NULL UNIQUE,
    payload_json TEXT NOT NULL
);
