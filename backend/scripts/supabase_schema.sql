-- =====================================================================
-- EduPath AI — Supabase Auth-Linked Schema (Alternative)
-- Team KRIYA | Meta Hackathon 2026
--
-- Purpose:
--   Alternative schema that uses Supabase Auth UUIDs directly as
--   primary keys. Includes auto-provisioning of student profiles
--   on signup via a database trigger on auth.users.
--
-- Difference from supabase_migration.sql:
--   - Uses UUID PKs referencing auth.users(id) instead of TEXT
--   - Uses JSONB columns instead of TEXT for structured data
--   - Includes per-user RLS policies (not just service-role bypass)
--   - Includes handle_new_user() trigger for auto-profile creation
--
-- Usage:
--   Run in Supabase SQL Editor. This is the auth-integrated variant.
-- =====================================================================


-- ─────────────────────────────────────────────────────────────────────
-- 1. STUDENTS — Core learner profiles (auth-linked)
--    Primary key references auth.users for seamless Supabase Auth
--    integration. JSONB columns enable native PostgreSQL queries.
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS students (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT DEFAULT '',
  email TEXT DEFAULT '',
  target_field TEXT DEFAULT 'tech',                    -- Learning domain
  learning_goal TEXT DEFAULT '',
  job_description TEXT DEFAULT '',
  weekly_hours INT DEFAULT 10,
  job_readiness_score FLOAT DEFAULT 0.0,               -- Composite metric (0.0–1.0)
  quiz_streak INT DEFAULT 0,                           -- Gamification: consecutive passes
  resume_skills JSONB DEFAULT '[]'::jsonb,             -- NLP-extracted resume skills
  self_assessed_skills JSONB DEFAULT '[]'::jsonb,      -- [{skill, level, proficiency}]
  jd_required_skills JSONB DEFAULT '[]'::jsonb,        -- Target JD skills
  completed_topics JSONB DEFAULT '[]'::jsonb,          -- Mastered topic IDs
  completed_projects JSONB DEFAULT '[]'::jsonb,        -- Submitted project IDs
  topics_studied JSONB DEFAULT '[]'::jsonb,            -- Topics with resource engagement
  clicked_resource_links JSONB DEFAULT '{}'::jsonb,    -- {topic_id: [urls]}
  badges JSONB DEFAULT '[]'::jsonb,                    -- Earned gamification badges
  onboarding_complete BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────────────
-- 2. STUDENT_QUIZZES — Quiz attempt records with adaptive difficulty
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_quizzes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id UUID REFERENCES students(id) ON DELETE CASCADE,
  topic_id TEXT NOT NULL,                              -- Curriculum topic identifier
  score FLOAT NOT NULL DEFAULT 0,                      -- Percentage 0–100
  total_questions INT DEFAULT 0,
  correct_answers INT DEFAULT 0,
  passed BOOLEAN DEFAULT FALSE,                         -- TRUE if score >= passing threshold
  difficulty TEXT DEFAULT 'medium',                     -- Adaptive: easy | medium | hard
  created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────────────
-- 3. STUDENT_PROJECTS — Project submissions with AI evaluation
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_projects (
  id TEXT PRIMARY KEY,
  student_id UUID REFERENCES students(id) ON DELETE CASCADE,
  project_title TEXT NOT NULL,
  project_type TEXT DEFAULT 'mini_project',             -- mini_project | capstone
  submission_text TEXT DEFAULT '',                      -- GitHub URL, code, or write-up
  score FLOAT DEFAULT 0,                               -- LLM evaluation score 0–100
  grade TEXT DEFAULT 'N/A',                             -- Letter grade A+ → F
  is_passing BOOLEAN DEFAULT FALSE,
  evaluation_data JSONB DEFAULT '{}'::jsonb,           -- Full AI evaluation payload
  submitted_at TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────────────
-- 4. STUDENT_ROADMAPS — Active learning roadmap (one per student)
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_roadmaps (
  student_id UUID PRIMARY KEY REFERENCES students(id) ON DELETE CASCADE,
  roadmap_data JSONB NOT NULL DEFAULT '{}'::jsonb,     -- Full LLM-generated roadmap
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────────────
-- 5. INDEXES — Optimise query performance
-- ─────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_quizzes_student ON student_quizzes(student_id);
CREATE INDEX IF NOT EXISTS idx_projects_student ON student_projects(student_id);


-- ─────────────────────────────────────────────────────────────────────
-- 6. ROW LEVEL SECURITY — Per-user isolation
--    Each user can only read/write their own data. The backend
--    service-role key bypasses RLS for admin operations.
-- ─────────────────────────────────────────────────────────────────────
ALTER TABLE students ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_quizzes ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_roadmaps ENABLE ROW LEVEL SECURITY;

-- Students table: full CRUD on own row
CREATE POLICY "Users read own data" ON students FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users update own data" ON students FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "Users insert own data" ON students FOR INSERT WITH CHECK (auth.uid() = id);

-- Quizzes: read and create own records
CREATE POLICY "Users read own quizzes" ON student_quizzes FOR SELECT USING (auth.uid() = student_id);
CREATE POLICY "Users insert own quizzes" ON student_quizzes FOR INSERT WITH CHECK (auth.uid() = student_id);

-- Projects: read and create own records
CREATE POLICY "Users read own projects" ON student_projects FOR SELECT USING (auth.uid() = student_id);
CREATE POLICY "Users insert own projects" ON student_projects FOR INSERT WITH CHECK (auth.uid() = student_id);

-- Roadmaps: full management of own roadmap
CREATE POLICY "Users read own roadmap" ON student_roadmaps FOR SELECT USING (auth.uid() = student_id);
CREATE POLICY "Users manage own roadmap" ON student_roadmaps FOR ALL USING (auth.uid() = student_id);


-- ─────────────────────────────────────────────────────────────────────
-- 7. TRIGGERS — Auto-update updated_at on row modification
-- ─────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER students_updated_at
  BEFORE UPDATE ON students
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER roadmaps_updated_at
  BEFORE UPDATE ON student_roadmaps
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();


-- ─────────────────────────────────────────────────────────────────────
-- 8. AUTO-PROVISIONING — Create student profile on auth signup
--    SECURITY DEFINER ensures the function runs with elevated
--    privileges to insert into the students table even before
--    the user's own RLS policies take effect.
-- ─────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO students (id, email, name)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name', '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Fire after each new auth.users row is created
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION handle_new_user();
