-- =====================================================================
-- EduPath AI — Enhanced Supabase Schema (12-Agent Pipeline)
-- Team KRIYA | Meta Hackathon 2026
--
-- Run this in the Supabase SQL Editor.
-- Includes the original 6 tables + 3 new tables for the
-- multi-agent pipeline. Idempotent — safe to run multiple times.
-- =====================================================================


-- ─────────────────────────────────────────────────────────────────────
-- 1. STUDENTS — Core learner profiles
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS students (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  email TEXT DEFAULT '',
  target_field TEXT DEFAULT 'tech',
  learning_goal TEXT DEFAULT '',
  job_description TEXT DEFAULT '',
  weekly_hours INT DEFAULT 10,
  job_readiness_score FLOAT DEFAULT 0.0,
  quiz_streak INT DEFAULT 0,
  resume_skills TEXT DEFAULT '[]',
  self_assessed_skills TEXT DEFAULT '[]',
  jd_required_skills TEXT DEFAULT '[]',
  completed_topics TEXT DEFAULT '[]',
  completed_projects TEXT DEFAULT '[]',
  topics_studied TEXT DEFAULT '[]',
  clicked_resource_links TEXT DEFAULT '{}',
  badges TEXT DEFAULT '[]',
  mastery_probabilities TEXT DEFAULT '{}',
  onboarding_complete BOOLEAN DEFAULT FALSE,
  current_roadmap_id UUID,
  total_roadmaps_completed INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);


-- ─────────────────────────────────────────────────────────────────────
-- 2. STUDENT_QUIZZES — Quiz attempt history
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_quizzes (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  student_id TEXT NOT NULL,
  topic_id TEXT NOT NULL DEFAULT '',
  score INT DEFAULT 0,
  total_questions INT DEFAULT 0,
  correct_answers INT DEFAULT 0,
  passed BOOLEAN DEFAULT FALSE,
  difficulty TEXT DEFAULT 'medium',
  created_at TIMESTAMPTZ DEFAULT now()
);


-- ─────────────────────────────────────────────────────────────────────
-- 3. STUDENT_PROJECTS — Project submissions & AI evaluations
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_projects (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
  student_id TEXT NOT NULL,
  project_title TEXT DEFAULT '',
  project_type TEXT DEFAULT 'mini_project',
  submission_text TEXT DEFAULT '',
  score INT DEFAULT 0,
  grade TEXT DEFAULT 'N/A',
  is_passing BOOLEAN DEFAULT FALSE,
  evaluation_data TEXT DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now()
);


-- ─────────────────────────────────────────────────────────────────────
-- 4. STUDENT_ROADMAPS — Active learning roadmap (1 per student)
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_roadmaps (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  student_id TEXT NOT NULL UNIQUE,
  roadmap_data TEXT DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);


-- ─────────────────────────────────────────────────────────────────────
-- 5. ROADMAP_HISTORY — Archived past roadmaps
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS roadmap_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  student_id TEXT NOT NULL,
  roadmap_data JSONB NOT NULL,
  topics_covered TEXT[] DEFAULT '{}',
  started_at TIMESTAMPTZ DEFAULT now(),
  archived_at TIMESTAMPTZ DEFAULT now(),
  completion_percentage FLOAT DEFAULT 0.0
);


-- ─────────────────────────────────────────────────────────────────────
-- 6. PROGRESS_SNAPSHOTS — Periodic progress analytics
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS progress_snapshots (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  student_id TEXT NOT NULL,
  snapshot_date DATE DEFAULT CURRENT_DATE,
  topics_completed TEXT[] DEFAULT '{}',
  quizzes_passed INT DEFAULT 0,
  projects_completed INT DEFAULT 0,
  job_readiness_score FLOAT DEFAULT 0.0,
  total_study_hours FLOAT DEFAULT 0.0
);


-- ═════════════════════════════════════════════════════════════════════
-- NEW TABLES for the 12-Agent Pipeline
-- ═════════════════════════════════════════════════════════════════════


-- ─────────────────────────────────────────────────────────────────────
-- 7. TRAJECTORY_MEMORY — Full pipeline state per student
--    Stores the complete inter-agent state: profile, council
--    proposals, section history, flags, and interventions.
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trajectory_memory (
  student_id TEXT PRIMARY KEY,
  student_profile JSONB DEFAULT '{}'::jsonb,
  domain_expert_proposal JSONB DEFAULT NULL,
  prereq_architect_proposal JSONB DEFAULT NULL,
  feasibility_proposal JSONB DEFAULT NULL,
  student_advocate_proposal JSONB DEFAULT NULL,
  conflict_resolution JSONB DEFAULT NULL,
  final_roadmap JSONB DEFAULT NULL,
  section_history JSONB DEFAULT '[]'::jsonb,
  quiz_attempts JSONB DEFAULT '{}'::jsonb,
  flags JSONB DEFAULT '[]'::jsonb,
  intervention_log JSONB DEFAULT '[]'::jsonb,
  bridging_topics_inserted INT DEFAULT 0,
  current_topic_materials JSONB DEFAULT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);


-- ─────────────────────────────────────────────────────────────────────
-- 8. LEARNING_MATERIALS — Cached Scout→Critic→Curator output
--    Keyed by (student_id, topic_id) to avoid re-running the
--    expensive 3-agent pipeline for the same topic.
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS learning_materials (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  student_id TEXT NOT NULL,
  topic_id TEXT NOT NULL,
  topic_name TEXT DEFAULT '',
  selected_course JSONB DEFAULT '{}'::jsonb,
  cheat_sheet JSONB DEFAULT '{}'::jsonb,
  study_notes JSONB DEFAULT '{}'::jsonb,
  mini_project JSONB DEFAULT '{}'::jsonb,
  scout_candidates_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(student_id, topic_id)
);


-- ─────────────────────────────────────────────────────────────────────
-- 9. AGENT_LOGS — Audit trail of all agent invocations
--    Excellent for hackathon demo: "watch our 12-agent debate"
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  student_id TEXT NOT NULL,
  agent_name TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT '',
  input_summary TEXT DEFAULT '',
  output_summary TEXT DEFAULT '',
  latency_ms INT DEFAULT 0,
  used_fallback BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT now()
);


-- ─────────────────────────────────────────────────────────────────────
-- INDEXES
-- ─────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_quizzes_student          ON student_quizzes(student_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_topic             ON student_quizzes(topic_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_created           ON student_quizzes(created_at);
CREATE INDEX IF NOT EXISTS idx_projects_student          ON student_projects(student_id);
CREATE INDEX IF NOT EXISTS idx_roadmaps_student          ON student_roadmaps(student_id);
CREATE INDEX IF NOT EXISTS idx_roadmap_history_student   ON roadmap_history(student_id);
CREATE INDEX IF NOT EXISTS idx_progress_student          ON progress_snapshots(student_id);
CREATE INDEX IF NOT EXISTS idx_progress_date             ON progress_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_trajectory_student        ON trajectory_memory(student_id);
CREATE INDEX IF NOT EXISTS idx_materials_student_topic   ON learning_materials(student_id, topic_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_student        ON agent_logs(student_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent          ON agent_logs(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_logs_created        ON agent_logs(created_at);


-- ─────────────────────────────────────────────────────────────────────
-- ROW LEVEL SECURITY — Open access (backend uses service key)
-- ─────────────────────────────────────────────────────────────────────
ALTER TABLE students           ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_quizzes    ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_projects   ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_roadmaps   ENABLE ROW LEVEL SECURITY;
ALTER TABLE roadmap_history    ENABLE ROW LEVEL SECURITY;
ALTER TABLE progress_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE trajectory_memory  ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_materials ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_logs         ENABLE ROW LEVEL SECURITY;

-- Allow all access (backend uses anon key for demo; production would use service key)
DO $$
DECLARE
  tbl TEXT;
BEGIN
  FOR tbl IN SELECT unnest(ARRAY[
    'students', 'student_quizzes', 'student_projects', 'student_roadmaps',
    'roadmap_history', 'progress_snapshots', 'trajectory_memory',
    'learning_materials', 'agent_logs'
  ]) LOOP
    EXECUTE format('DROP POLICY IF EXISTS "Allow full access" ON %I', tbl);
    EXECUTE format('CREATE POLICY "Allow full access" ON %I FOR ALL USING (true) WITH CHECK (true)', tbl);
  END LOOP;
END $$;


-- ─────────────────────────────────────────────────────────────────────
-- TRIGGERS — Auto-update updated_at
-- ─────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_students_updated_at') THEN
    CREATE TRIGGER set_students_updated_at
      BEFORE UPDATE ON students
      FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_roadmaps_updated_at') THEN
    CREATE TRIGGER set_roadmaps_updated_at
      BEFORE UPDATE ON student_roadmaps
      FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_trajectory_updated_at') THEN
    CREATE TRIGGER set_trajectory_updated_at
      BEFORE UPDATE ON trajectory_memory
      FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
  END IF;
END $$;


-- =====================================================================
-- Migration complete. 9 tables provisioned.
-- =====================================================================
