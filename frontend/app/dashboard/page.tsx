/**
 * EduPath AI — Student Dashboard
 * Team KRIYA | Meta Hackathon 2026
 *
 * Central hub for the student experience: roadmap view, adaptive quizzes,
 * project submissions with AI evaluation, badge gallery, career readiness
 * tracking, and profile analytics. Communicates with the FastAPI backend
 * via the api.ts client.
 */

"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { apiGet, apiPost } from "@/lib/api";
import { supabase } from "@/lib/supabase";

type Tab = "roadmap" | "quizzes" | "badges" | "career" | "profile";

interface StudentData {
  name: string;
  target_field: string;
  learning_goal: string;
  completed_topics: string[];
  quiz_history: any[];
  completed_projects: string[];
  badges: any[];
  job_readiness_score: number;
  weekly_hours: number;
  quiz_streak: number;
  topics_studied: string[];
}

export default function DashboardPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<Tab>("roadmap");
  const [studentId, setStudentId] = useState<string>("");
  const [student, setStudent] = useState<StudentData | null>(null);
  const [roadmap, setRoadmap] = useState<any>(null);
  const [quiz, setQuiz] = useState<any>(null);
  const [quizAnswers, setQuizAnswers] = useState<number[]>([]);
  const [quizResult, setQuizResult] = useState<any>(null);
  const [badges, setBadges] = useState<any>(null);
  const [career, setCareer] = useState<any>(null);
  const [events, setEvents] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Profile state
  const [profileData, setProfileData] = useState<any>(null);
  const [roadmapHistory, setRoadmapHistory] = useState<any[]>([]);
  const [progressData, setProgressData] = useState<any>(null);

  // Project submission state
  const [projectModal, setProjectModal] = useState<any>(null);
  const [projectSubmission, setProjectSubmission] = useState("");
  const [projectReport, setProjectReport] = useState<any>(null);
  const [submittingProject, setSubmittingProject] = useState(false);

  // Quit roadmap confirmation
  const [showQuitConfirm, setShowQuitConfirm] = useState(false);
  const [archiving, setArchiving] = useState(false);

  useEffect(() => {
    // Check Supabase auth first
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        router.push("/auth");
        return;
      }
      const id = session.user.id;
      setStudentId(id);
      localStorage.setItem("edupath_student_id", id);
      loadStudentData(id);
    });
  }, [router]);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    localStorage.removeItem("edupath_student_id");
    localStorage.removeItem("edupath_student_name");
    router.push("/");
  };

  const loadStudentData = async (id: string) => {
    try {
      const data = await apiGet(`/api/onboarding/profile/${id}`);
      setStudent(data);
    } catch (err) {
      console.error("Failed to load student data, redirecting to onboarding");
      router.push("/onboarding");
    }
  };

  const generateRoadmap = async () => {
    setLoading(true);
    try {
      const data = await apiPost("/api/roadmap/generate", { student_id: studentId, force_regenerate: true });
      setRoadmap(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadRoadmap = async () => {
    try {
      const data = await apiGet(`/api/roadmap/${studentId}`);
      setRoadmap(data);
    } catch {
      // No cached roadmap
    }
  };

  const handleQuitRoadmap = async () => {
    setArchiving(true);
    try {
      await apiPost("/api/roadmap/archive", { student_id: studentId });
      setRoadmap(null);
      setShowQuitConfirm(false);
      // Refresh student data to get updated completed_topics
      loadStudentData(studentId);
    } catch (err: any) {
      setError(err.message || "Failed to archive roadmap");
    } finally {
      setArchiving(false);
    }
  };

  const handleEditRoadmap = () => {
    router.push("/onboarding?edit=true");
  };

  const generateQuiz = async (topicId: string) => {
    setLoading(true);
    setQuizResult(null);
    setQuizAnswers([]);
    try {
      const data = await apiPost("/api/quiz/generate", { student_id: studentId, topic_id: topicId });
      setQuiz(data);
      setQuizAnswers(new Array(data.questions?.length || 0).fill(-1));
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const submitQuiz = async () => {
    if (!quiz) return;
    setLoading(true);
    try {
      const data = await apiPost("/api/quiz/submit", {
        student_id: studentId,
        topic_id: quiz.topic_id,
        questions: quiz.questions,
        answers: quizAnswers,
      });
      setQuizResult(data);
      loadStudentData(studentId);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadBadges = async () => {
    try {
      const data = await apiGet(`/api/badges/${studentId}`);
      setBadges(data);
    } catch (err) {
      console.error("Failed to load badges");
    }
  };

  const loadCareer = async () => {
    try {
      const [readiness, evts] = await Promise.all([
        apiGet(`/api/career/readiness/${studentId}`),
        apiGet(`/api/career/events/${studentId}`),
      ]);
      setCareer(readiness);
      setEvents(evts);
    } catch (err) {
      console.error("Failed to load career data");
    }
  };

  const loadProfile = async () => {
    try {
      const [profile, history, progress] = await Promise.all([
        apiGet(`/api/profile/${studentId}`),
        apiGet(`/api/profile/${studentId}/history`),
        apiGet(`/api/profile/${studentId}/progress`),
      ]);
      setProfileData(profile);
      setRoadmapHistory(history?.history || []);
      setProgressData(progress);
    } catch (err) {
      console.error("Failed to load profile data");
    }
  };

  const submitProject = async () => {
    if (!projectModal || !projectSubmission.trim()) return;
    setSubmittingProject(true);
    try {
      const data = await apiPost("/api/projects/submit", {
        student_id: studentId,
        project_title: projectModal.title,
        project_description: projectModal.description || "",
        project_type: projectModal.type || "mini_project",
        submission_text: projectSubmission,
        requirements: projectModal.requirements || [],
      });
      setProjectReport(data.evaluation);
      loadStudentData(studentId);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSubmittingProject(false);
    }
  };

  useEffect(() => {
    if (!studentId) return;
    if (activeTab === "roadmap") loadRoadmap();
    if (activeTab === "badges") loadBadges();
    if (activeTab === "career") loadCareer();
    if (activeTab === "profile") loadProfile();
  }, [activeTab, studentId]);

  // Helper: Convert a skill name to a URL-safe topic ID
  const getTopicIdForWeek = (week: any): string => {
    const raw = week.skillsCovered?.[0] || week.title || "unknown";
    return encodeURIComponent(raw.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, '_'));
  };

  const TABS: { id: Tab; icon: string; label: string }[] = [
    { id: "roadmap", icon: "🗺️", label: "Roadmap" },
    { id: "quizzes", icon: "📝", label: "Quizzes" },
    { id: "badges", icon: "🏆", label: "Badges" },
    { id: "career", icon: "💼", label: "Career" },
    { id: "profile", icon: "👤", label: "Profile" },
  ];

  if (!studentId) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--bg-primary)" }}>
        <div className="glass-card p-8 text-center max-w-md">
          <div className="text-4xl mb-4">🎓</div>
          <h2 className="text-2xl font-bold mb-2">Welcome to EduPath AI</h2>
          <p className="mb-6" style={{ color: "var(--text-secondary)" }}>Complete onboarding to access your dashboard.</p>
          <Link href="/onboarding" className="glow-btn inline-block">Start Onboarding →</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex" style={{ background: "var(--bg-primary)" }}>
      {/* ── Sidebar ── */}
      <aside className="w-64 p-4 flex flex-col gap-2 fixed h-full" style={{ background: "var(--bg-secondary)", borderRight: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2 px-4 py-3 mb-4">
          <span className="text-xl">🎓</span>
          <span className="font-bold gradient-text">EduPath AI</span>
        </div>

        {student && (
          <div className="px-4 py-3 rounded-xl mb-4" style={{ background: "var(--bg-card)" }}>
            <div className="text-sm font-bold">{student.name || "Student"}</div>
            <div className="text-xs" style={{ color: "var(--text-secondary)" }}>{student.target_field?.toUpperCase()} • {student.weekly_hours}h/week</div>
            <div className="mt-2">
              <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Job Readiness</div>
              <div className="progress-bar mt-1">
                <div className="progress-fill" style={{ width: `${(student.job_readiness_score || 0) * 100}%` }} />
              </div>
              <div className="text-xs font-bold mt-1 gradient-text">{Math.round((student.job_readiness_score || 0) * 100)}%</div>
            </div>
          </div>
        )}

        {TABS.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`sidebar-item flex items-center gap-3 text-sm ${activeTab === tab.id ? "sidebar-item-active" : ""}`}>
            <span>{tab.icon}</span> {tab.label}
          </button>
        ))}

        <div className="mt-auto space-y-1">
          <Link href="/" className="sidebar-item flex items-center gap-3 text-sm">
            <span>🏠</span> Home
          </Link>
          <button onClick={handleLogout} className="sidebar-item flex items-center gap-3 text-sm w-full text-left" style={{ color: "var(--accent-red)" }}>
            <span>🚪</span> Logout
          </button>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="flex-1 ml-64 p-8">
        {error && (
          <div className="mb-4 px-4 py-2 rounded-lg" style={{ background: "rgba(239,68,68,0.1)", color: "var(--accent-red)" }}>
            {error}
            <button onClick={() => setError("")} className="ml-4 font-bold">×</button>
          </div>
        )}

        {/* ══════ QUIT ROADMAP CONFIRMATION MODAL ══════ */}
        {showQuitConfirm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.7)" }}>
            <div className="glass-card p-8 max-w-md w-full mx-4 text-center">
              <div className="text-4xl mb-4">⚠️</div>
              <h2 className="text-xl font-bold mb-2">Quit Current Roadmap?</h2>
              <p className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>
                Your progress will be saved to your profile history. You can start a new roadmap anytime.
              </p>
              <div className="flex gap-3 justify-center">
                <button onClick={() => setShowQuitConfirm(false)}
                  className="px-6 py-2 rounded-lg text-sm font-bold"
                  style={{ border: "1px solid var(--border)", color: "var(--text-secondary)" }}>
                  Cancel
                </button>
                <button onClick={handleQuitRoadmap} disabled={archiving}
                  className="px-6 py-2 rounded-lg text-sm font-bold"
                  style={{ background: "rgba(239,68,68,0.2)", color: "var(--accent-red)", border: "1px solid rgba(239,68,68,0.3)" }}>
                  {archiving ? "Archiving..." : "🚪 Quit & Archive"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ══════ PROJECT SUBMISSION MODAL ══════ */}
        {projectModal && !projectReport && (
          <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.7)" }}>
            <div className="glass-card p-8 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-xl font-bold">📤 Submit Project</h2>
                  <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{projectModal.title}</p>
                </div>
                <button onClick={() => { setProjectModal(null); setProjectSubmission(""); }} className="text-2xl">×</button>
              </div>

              {projectModal.description && (
                <div className="p-3 rounded-lg mb-4" style={{ background: "var(--bg-primary)" }}>
                  <div className="text-xs font-bold mb-1" style={{ color: "var(--text-secondary)" }}>Project Description</div>
                  <p className="text-sm">{projectModal.description}</p>
                </div>
              )}

              {projectModal.requirements?.length > 0 && (
                <div className="p-3 rounded-lg mb-4" style={{ background: "var(--bg-primary)" }}>
                  <div className="text-xs font-bold mb-1" style={{ color: "var(--text-secondary)" }}>Requirements</div>
                  <ul className="text-sm space-y-1">
                    {projectModal.requirements.map((r: string, i: number) => (
                      <li key={i}>• {r}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="mb-4">
                <label className="text-sm font-bold block mb-2">Paste your GitHub repo URL or project files/code:</label>
                <textarea
                  className="w-full p-4 rounded-lg text-sm font-mono"
                  style={{ background: "var(--bg-primary)", border: "1px solid var(--border)", color: "var(--text-primary)", minHeight: "200px" }}
                  placeholder={"https://github.com/yourusername/your-project\n\nOR paste your code / project description here...\n\nInclude:\n- What you built\n- Key features\n- Technologies used\n- How to run it"}
                  value={projectSubmission}
                  onChange={(e) => setProjectSubmission(e.target.value)}
                />
              </div>

              <button
                onClick={submitProject}
                disabled={!projectSubmission.trim() || submittingProject}
                className="glow-btn w-full disabled:opacity-50"
              >
                {submittingProject ? "🤖 AI is evaluating your project..." : "Submit for AI Evaluation →"}
              </button>
            </div>
          </div>
        )}

        {/* ══════ PROJECT REPORT MODAL ══════ */}
        {projectReport && (
          <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.7)" }}>
            <div className="glass-card p-8 max-w-3xl w-full mx-4 max-h-[85vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold">📊 AI Evaluation Report</h2>
                <button onClick={() => { setProjectReport(null); setProjectModal(null); setProjectSubmission(""); }} className="text-2xl">×</button>
              </div>

              {/* Score & Grade */}
              <div className="flex items-center gap-6 mb-6">
                <div className="text-center">
                  <div className="text-5xl font-bold gradient-text">{projectReport.score}%</div>
                  <div className="text-sm" style={{ color: "var(--text-secondary)" }}>Score</div>
                </div>
                <div className="text-center">
                  <div className="text-5xl font-bold" style={{ color: projectReport.is_passing ? "var(--accent-green)" : "var(--accent-red)" }}>
                    {projectReport.grade}
                  </div>
                  <div className="text-sm" style={{ color: "var(--text-secondary)" }}>Grade</div>
                </div>
                <div className="flex-1 p-4 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                  <p className="text-sm">{projectReport.overall_feedback}</p>
                </div>
              </div>

              {/* Technical Analysis */}
              {projectReport.technical_analysis && (
                <div className="mb-6">
                  <h3 className="font-bold mb-3">📋 Technical Analysis</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {Object.entries(projectReport.technical_analysis).map(([key, val]: [string, any]) => (
                      <div key={key} className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-bold capitalize">{key.replace(/_/g, " ")}</span>
                          <span className="text-xs font-bold" style={{ color: val.score >= 7 ? "var(--accent-green)" : val.score >= 5 ? "var(--accent-amber)" : "var(--accent-red)" }}>
                            {val.score}/10
                          </span>
                        </div>
                        <div className="progress-bar h-1.5 mb-1">
                          <div className="progress-fill" style={{ width: `${val.score * 10}%` }} />
                        </div>
                        <p className="text-[10px]" style={{ color: "var(--text-secondary)" }}>{val.feedback}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Strengths & Improvements */}
              <div className="grid md:grid-cols-2 gap-4 mb-6">
                <div className="p-4 rounded-lg" style={{ background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.2)" }}>
                  <h4 className="text-sm font-bold mb-2" style={{ color: "var(--accent-green)" }}>✅ Strengths</h4>
                  <ul className="text-xs space-y-1">
                    {projectReport.strengths?.map((s: string, i: number) => (
                      <li key={i}>• {s}</li>
                    ))}
                  </ul>
                </div>
                <div className="p-4 rounded-lg" style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.2)" }}>
                  <h4 className="text-sm font-bold mb-2" style={{ color: "var(--accent-amber)" }}>🔧 Improvements</h4>
                  <ul className="text-xs space-y-1">
                    {projectReport.improvements?.map((s: string, i: number) => (
                      <li key={i}>• {s}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Next Steps */}
              {projectReport.next_steps && (
                <div className="p-4 rounded-lg mb-6" style={{ background: "var(--bg-primary)" }}>
                  <h4 className="text-sm font-bold mb-2">🚀 Next Steps</h4>
                  <ul className="text-xs space-y-1">
                    {projectReport.next_steps.map((s: string, i: number) => (
                      <li key={i}>→ {s}</li>
                    ))}
                  </ul>
                </div>
              )}

              <button onClick={() => { setProjectReport(null); setProjectModal(null); setProjectSubmission(""); }}
                className="glow-btn w-full">
                Close Report
              </button>
            </div>
          </div>
        )}

        {/* ═══ ROADMAP TAB ═══ */}
        {activeTab === "roadmap" && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold">🗺️ Your Learning Roadmap</h1>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Personalized path generated by AI</p>
              </div>
              <div className="flex items-center gap-3">
                {roadmap && (
                  <>
                    <button onClick={handleEditRoadmap}
                      className="px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-all hover:scale-105"
                      style={{ background: "rgba(139,92,246,0.15)", color: "var(--accent-purple)", border: "1px solid rgba(139,92,246,0.3)" }}>
                      ✏️ Edit Roadmap
                    </button>
                    <button onClick={() => setShowQuitConfirm(true)}
                      className="px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-2 transition-all hover:scale-105"
                      style={{ background: "rgba(239,68,68,0.1)", color: "var(--accent-red)", border: "1px solid rgba(239,68,68,0.2)" }}>
                      🚪 Quit Roadmap
                    </button>
                  </>
                )}
                <button onClick={generateRoadmap} disabled={loading} className="glow-btn !text-sm disabled:opacity-50">
                  {loading ? "Generating..." : roadmap ? "🔄 Regenerate" : "✨ Generate Roadmap"}
                </button>
              </div>
            </div>

            {/* Roadmap progress banner */}
            {roadmap?.weeks && student && (
              <div className="glass-card p-4 mb-6 flex items-center gap-6" style={{ border: "1px solid rgba(79,110,247,0.3)" }}>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-bold">Roadmap Progress</span>
                    <span className="text-sm font-bold gradient-text">
                      {Math.round(((student.completed_topics?.length || 0) / Math.max(roadmap.weeks.length, 1)) * 100)}%
                    </span>
                  </div>
                  <div className="progress-bar h-2.5">
                    <div className="progress-fill" style={{ width: `${((student.completed_topics?.length || 0) / Math.max(roadmap.weeks.length, 1)) * 100}%` }} />
                  </div>
                  <div className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
                    {student.completed_topics?.length || 0} of {roadmap.weeks.length} topics completed
                  </div>
                </div>
                <div className="text-center px-4" style={{ borderLeft: "1px solid var(--border)" }}>
                  <div className="text-2xl font-bold gradient-text">🔥 {student.quiz_streak || 0}</div>
                  <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Streak</div>
                </div>
              </div>
            )}

            {roadmap?.weeks ? (
              <div>
                <div className="space-y-4">
                  {roadmap.weeks.map((week: any, i: number) => {
                    const topicId = getTopicIdForWeek(week);
                    const isCompleted = student?.completed_topics?.some(t =>
                      t.toLowerCase().replace(/\s+/g, '_') === decodeURIComponent(topicId).toLowerCase()
                    );
                    return (
                      <div key={i} className="glass-card p-6 relative" style={{
                        animationDelay: `${i * 0.05}s`,
                        borderColor: isCompleted ? "rgba(34,197,94,0.4)" : undefined,
                      }}>
                        {isCompleted && (
                          <div className="absolute top-3 right-3 px-2 py-0.5 rounded-full text-[10px] font-bold"
                            style={{ background: "rgba(34,197,94,0.2)", color: "var(--accent-green)" }}>
                            ✓ Completed
                          </div>
                        )}
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <span className="px-2 py-0.5 rounded-full text-xs font-bold" style={{ background: "rgba(79,110,247,0.2)", color: "var(--accent-blue)" }}>
                                Week {week.weekNumber || i + 1}
                              </span>
                              <h3 className="font-bold">{week.title}</h3>
                            </div>
                            {week.learningObjectives && (
                              <ul className="text-sm space-y-1 ml-4" style={{ color: "var(--text-secondary)" }}>
                                {week.learningObjectives.map((obj: string, j: number) => (
                                  <li key={j}>• {obj}</li>
                                ))}
                              </ul>
                            )}
                            {week.skillsCovered && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {week.skillsCovered.map((skill: string, j: number) => (
                                  <span key={j} className="px-2 py-0.5 rounded text-[10px]" style={{ background: "rgba(139,92,246,0.15)", color: "var(--accent-purple)" }}>
                                    {skill}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                          <div className="flex flex-col gap-2 ml-4">
                            <Link href={`/dashboard/topic?id=${topicId}`}
                              className="text-xs px-3 py-1.5 rounded-lg font-bold text-center whitespace-nowrap" style={{ background: "rgba(79,110,247,0.15)", color: "var(--accent-blue)" }}>
                              📚 Study Topic
                            </Link>
                            <button onClick={() => { setActiveTab("quizzes"); generateQuiz(decodeURIComponent(topicId)); }}
                              className="text-xs px-3 py-1.5 rounded-lg font-bold whitespace-nowrap" style={{ background: "rgba(34,197,94,0.15)", color: "var(--accent-green)" }}>
                              📝 Take Quiz
                            </button>
                          </div>
                        </div>

                        {/* Mini Project */}
                        {week.mini_project && (
                          <div className="mt-3 p-3 rounded-lg" style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.2)" }}>
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-xs font-bold" style={{ color: "var(--accent-amber)" }}>🛠️ Mini Project: {week.mini_project.title}</div>
                                <div className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>{week.mini_project.description}</div>
                              </div>
                              <button
                                onClick={() => setProjectModal({
                                  title: week.mini_project.title,
                                  description: week.mini_project.description,
                                  type: "mini_project",
                                  requirements: week.mini_project.requirements || [],
                                })}
                                className="text-xs px-3 py-1.5 rounded-lg font-bold whitespace-nowrap ml-3"
                                style={{ background: "rgba(245,158,11,0.2)", color: "var(--accent-amber)" }}
                              >
                                📤 Submit
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* ── Capstone Projects Section ── */}
                {roadmap.capstone_projects && roadmap.capstone_projects.length > 0 && (
                  <div className="mt-10">
                    <div className="flex items-center gap-3 mb-6">
                      <span className="text-2xl">🏆</span>
                      <div>
                        <h2 className="text-xl font-bold">Capstone Projects</h2>
                        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Complete these to demonstrate mastery & earn your final badges</p>
                      </div>
                    </div>
                    <div className="grid md:grid-cols-2 gap-6">
                      {roadmap.capstone_projects.map((cap: any, i: number) => (
                        <div key={i} className="glass-card p-6 relative overflow-hidden" style={{ border: "1px solid rgba(245,158,11,0.3)" }}>
                          <div className="absolute top-0 right-0 px-3 py-1 rounded-bl-lg text-[10px] font-bold" style={{ background: "rgba(245,158,11,0.2)", color: "var(--accent-amber)" }}>
                            CAPSTONE {i + 1}
                          </div>
                          <h3 className="font-bold text-lg mb-2 mt-2">{cap.title}</h3>
                          <p className="text-sm mb-3" style={{ color: "var(--text-secondary)" }}>{cap.description}</p>
                          {cap.expected_output && (
                            <div className="text-xs mb-3 p-2 rounded" style={{ background: "var(--bg-primary)" }}>
                              <span className="font-bold">Expected Output:</span> {cap.expected_output}
                            </div>
                          )}
                          {cap.requirements && (
                            <div className="flex flex-wrap gap-1 mb-3">
                              {cap.requirements.map((req: string, j: number) => (
                                <span key={j} className="px-2 py-0.5 rounded text-[10px]" style={{ background: "rgba(139,92,246,0.15)", color: "var(--accent-purple)" }}>
                                  {req}
                                </span>
                              ))}
                            </div>
                          )}
                          <button
                            onClick={() => setProjectModal({
                              title: cap.title,
                              description: cap.description,
                              type: "capstone",
                              requirements: cap.requirements || [],
                            })}
                            className="glow-btn !text-sm w-full mt-2"
                          >
                            📤 Submit Capstone Project
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="glass-card p-12 text-center">
                <div className="text-4xl mb-4 animate-float">🗺️</div>
                <h3 className="text-lg font-bold mb-2">No Roadmap Yet</h3>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Click &quot;Generate Roadmap&quot; to create your personalized learning path.</p>
              </div>
            )}
          </div>
        )}

        {/* ═══ QUIZZES TAB ═══ */}
        {activeTab === "quizzes" && (
          <div>
            <h1 className="text-2xl font-bold mb-6">📝 Adaptive Quizzes</h1>

            {!quiz && !quizResult && (
              <div className="glass-card p-8 text-center">
                <div className="text-4xl mb-4">📝</div>
                <h3 className="text-lg font-bold mb-2">Take a Quiz</h3>
                <p className="text-sm mb-4" style={{ color: "var(--text-secondary)" }}>Select a topic from your roadmap, or choose a common one below.</p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {["python_basics", "data_structures", "machine_learning", "flask_framework", "api_design", "web_development"].map((t) => (
                    <button key={t} onClick={() => generateQuiz(t)}
                      className="px-4 py-2 rounded-lg text-sm transition-all hover:scale-105"
                      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
                      {t.replace(/_/g, " ")}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {quiz && !quizResult && (
              <div className="space-y-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold">{quiz.topic_name}</h3>
                  <span className="px-3 py-1 rounded-full text-xs" style={{ background: "rgba(79,110,247,0.15)", color: "var(--accent-blue)" }}>
                    {quiz.difficulty?.toUpperCase()}
                  </span>
                </div>

                {quiz.questions?.map((q: any, i: number) => (
                  <div key={i} className="glass-card p-6">
                    <h4 className="font-medium mb-3">Q{i + 1}. {q.question}</h4>
                    <div className="space-y-2">
                      {q.options?.map((opt: string, j: number) => (
                        <button key={j} onClick={() => {
                          const newAnswers = [...quizAnswers];
                          newAnswers[i] = j;
                          setQuizAnswers(newAnswers);
                        }}
                          className="w-full text-left px-4 py-3 rounded-lg text-sm transition-all"
                          style={{
                            background: quizAnswers[i] === j ? "rgba(79,110,247,0.2)" : "var(--bg-primary)",
                            border: `1px solid ${quizAnswers[i] === j ? "var(--accent-blue)" : "var(--border)"}`,
                          }}>
                          {opt}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}

                <button onClick={submitQuiz} disabled={quizAnswers.includes(-1) || loading}
                  className="glow-btn w-full disabled:opacity-50">
                  {loading ? "Scoring..." : "Submit Answers"}
                </button>
              </div>
            )}

            {quizResult && (
              <div className="glass-card p-8 text-center">
                <div className="text-5xl mb-4">{quizResult.passed ? "🎉" : quizResult.score >= 50 ? "📖" : "💪"}</div>
                <h3 className="text-2xl font-bold mb-2">
                  Score: <span className={quizResult.passed ? "text-green-400" : quizResult.score >= 50 ? "text-amber-400" : "text-red-400"}>
                    {quizResult.score}%
                  </span>
                </h3>
                <p className="mb-4" style={{ color: "var(--text-secondary)" }}>{quizResult.message}</p>
                <div className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>
                  {quizResult.correct_answers}/{quizResult.total_questions} correct
                </div>

                <div className="mb-6 p-4 rounded-xl" style={{ background: "var(--bg-primary)", border: "1px solid var(--border)" }}>
                  <h4 className="text-sm font-bold mb-2">📥 Download Study Resources</h4>
                  <div className="flex flex-wrap gap-2 justify-center">
                    <a href={`https://www.google.com/search?q=${quiz?.topic_name?.replace(/ /g, '+')}+cheat+sheet+pdf`}
                      target="_blank" rel="noopener noreferrer"
                      className="px-4 py-2 rounded-lg text-xs font-bold transition-all hover:scale-105"
                      style={{ background: "rgba(79,110,247,0.15)", color: "var(--accent-blue)" }}>
                      📄 Cheat Sheet
                    </a>
                    <a href={`https://www.google.com/search?q=${quiz?.topic_name?.replace(/ /g, '+')}+study+notes+pdf`}
                      target="_blank" rel="noopener noreferrer"
                      className="px-4 py-2 rounded-lg text-xs font-bold transition-all hover:scale-105"
                      style={{ background: "rgba(139,92,246,0.15)", color: "var(--accent-purple)" }}>
                      📝 Study Notes
                    </a>
                  </div>
                </div>

                <div className="flex gap-3 justify-center">
                  <button onClick={() => { setQuiz(null); setQuizResult(null); }} className="glow-btn !text-sm">
                    Take Another Quiz
                  </button>
                  {!quizResult.passed && (
                    <button onClick={() => { setQuizResult(null); setQuizAnswers(new Array(quiz?.questions?.length || 0).fill(-1)); }}
                      className="px-6 py-2 rounded-lg text-sm" style={{ border: "1px solid var(--border)" }}>
                      Retry
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══ BADGES TAB ═══ */}
        {activeTab === "badges" && (
          <div>
            <h1 className="text-2xl font-bold mb-2">🏆 Badges & Achievements</h1>
            <p className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>Collect them all to prove your mastery!</p>
            {badges ? (
              <div>
                <div className="glass-card p-4 mb-6 flex items-center justify-between">
                  <div>
                    <span className="text-2xl font-bold gradient-text">{badges.earned_count}</span>
                    <span className="text-sm ml-2" style={{ color: "var(--text-secondary)" }}>/ {badges.total_count} badges earned</span>
                  </div>
                  <div className="progress-bar w-48">
                    <div className="progress-fill" style={{ width: `${(badges.earned_count / badges.total_count) * 100}%` }} />
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {badges.badges?.map((badge: any, i: number) => (
                    <div key={i} className={`p-4 rounded-2xl text-center transition-all hover:scale-105 ${badge.earned ? "badge-earned" : "badge-locked"}`}>
                      <div className="text-3xl mb-2">{badge.icon}</div>
                      <h4 className="text-sm font-bold">{badge.name}</h4>
                      <p className="text-[10px] mt-1" style={{ color: "var(--text-secondary)" }}>{badge.description}</p>
                      {badge.earned && <span className="inline-block mt-2 text-[10px] font-bold" style={{ color: "var(--accent-green)" }}>✓ Earned</span>}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="glass-card p-8 text-center">
                <div className="text-4xl mb-4 animate-float">🏆</div>
                <p style={{ color: "var(--text-secondary)" }}>Loading badges...</p>
              </div>
            )}
          </div>
        )}

        {/* ═══ CAREER TAB ═══ */}
        {activeTab === "career" && (
          <div>
            <h1 className="text-2xl font-bold mb-6">💼 Job Readiness & Career</h1>

            {career && (
              <div className="grid md:grid-cols-2 gap-6 mb-8">
                <div className="glass-card p-6">
                  <h3 className="font-bold mb-4">Job Readiness Score</h3>
                  <div className="text-center">
                    <div className="text-5xl font-bold gradient-text">{Math.round(career.job_readiness_score * 100)}%</div>
                    <div className="progress-bar mt-4 h-3">
                      <div className="progress-fill" style={{ width: `${career.job_readiness_score * 100}%` }} />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3 mt-6 text-sm">
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div style={{ color: "var(--text-secondary)" }}>Topics</div>
                      <div className="font-bold">{career.breakdown?.topics_completed || 0}</div>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div style={{ color: "var(--text-secondary)" }}>Quiz Avg</div>
                      <div className="font-bold">{career.breakdown?.average_quiz_score || 0}%</div>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div style={{ color: "var(--text-secondary)" }}>Capstones</div>
                      <div className="font-bold">{career.breakdown?.projects_completed || 0}</div>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div style={{ color: "var(--text-secondary)" }}>Streak</div>
                      <div className="font-bold">🔥 {career.breakdown?.quiz_streak || 0}</div>
                    </div>
                  </div>
                </div>

                <div className="glass-card p-6">
                  <h3 className="font-bold mb-4">Interview Prep</h3>
                  {career.interview_prep_unlocked ? (
                    <div className="text-center p-6">
                      <div className="text-3xl mb-2">🎯</div>
                      <p className="font-bold text-green-400">Unlocked!</p>
                      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>You&apos;re ready for interview preparation.</p>
                    </div>
                  ) : (
                    <div className="text-center p-6">
                      <div className="text-3xl mb-2 opacity-50">🔒</div>
                      <p className="font-bold" style={{ color: "var(--text-secondary)" }}>Locked</p>
                      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>Reach 70% job readiness to unlock.</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {events && (
              <div>
                <h3 className="font-bold mb-4">🎪 Recommended Events & Opportunities</h3>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {events.events?.map((event: any, i: number) => (
                    <a key={i} href={event.url} target="_blank" rel="noopener noreferrer" className="glass-card p-4 block hover:scale-[1.02] transition-transform">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold" style={{
                          background: event.type === "hackathon" ? "rgba(139,92,246,0.2)" : event.type === "conference" ? "rgba(79,110,247,0.2)" : "rgba(34,197,94,0.2)",
                          color: event.type === "hackathon" ? "var(--accent-purple)" : event.type === "conference" ? "var(--accent-blue)" : "var(--accent-green)",
                        }}>{event.type}</span>
                      </div>
                      <h4 className="font-bold text-sm">{event.name}</h4>
                      <p className="text-[10px] mt-1" style={{ color: "var(--text-secondary)" }}>{event.url}</p>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══ PROFILE TAB ═══ */}
        {activeTab === "profile" && (
          <div>
            <h1 className="text-2xl font-bold mb-6">👤 Your Profile</h1>

            {/* Profile Overview */}
            {student && (
              <div className="grid md:grid-cols-3 gap-6 mb-8">
                <div className="glass-card p-6 md:col-span-2">
                  <h3 className="font-bold mb-4">📋 Student Information</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Name</div>
                      <div className="font-bold">{student.name || "—"}</div>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Field</div>
                      <div className="font-bold">{student.target_field?.toUpperCase() || "—"}</div>
                    </div>
                    <div className="p-3 rounded-lg col-span-2" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Learning Goal</div>
                      <div className="font-bold">{student.learning_goal || "—"}</div>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Weekly Hours</div>
                      <div className="font-bold">{student.weekly_hours}h / week</div>
                    </div>
                    <div className="p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Quiz Streak</div>
                      <div className="font-bold">🔥 {student.quiz_streak || 0}</div>
                    </div>
                  </div>
                </div>

                <div className="glass-card p-6">
                  <h3 className="font-bold mb-4">📊 Stats</h3>
                  <div className="space-y-4">
                    <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-3xl font-bold gradient-text">{student.completed_topics?.length || 0}</div>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Topics Completed</div>
                    </div>
                    <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-3xl font-bold gradient-text">{profileData?.stats?.total_quizzes || 0}</div>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Quizzes Taken</div>
                    </div>
                    <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-3xl font-bold gradient-text">{profileData?.stats?.total_projects || 0}</div>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Projects Submitted</div>
                    </div>
                    <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                      <div className="text-3xl font-bold gradient-text">{Math.round((student.job_readiness_score || 0) * 100)}%</div>
                      <div className="text-xs" style={{ color: "var(--text-secondary)" }}>Job Readiness</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Topics Covered */}
            {student?.completed_topics && student.completed_topics.length > 0 && (
              <div className="glass-card p-6 mb-8">
                <h3 className="font-bold mb-4">✅ Topics You&apos;ve Covered</h3>
                <div className="flex flex-wrap gap-2">
                  {student.completed_topics.map((topic, i) => (
                    <span key={i} className="px-3 py-1.5 rounded-lg text-xs font-bold"
                      style={{ background: "rgba(34,197,94,0.15)", color: "var(--accent-green)", border: "1px solid rgba(34,197,94,0.3)" }}>
                      ✓ {topic.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Topics Studied (clicked but not completed) */}
            {student?.topics_studied && student.topics_studied.length > 0 && (
              <div className="glass-card p-6 mb-8">
                <h3 className="font-bold mb-4">📚 Topics Studied</h3>
                <div className="flex flex-wrap gap-2">
                  {student.topics_studied.map((topic, i) => (
                    <span key={i} className="px-3 py-1.5 rounded-lg text-xs font-bold"
                      style={{ background: "rgba(79,110,247,0.15)", color: "var(--accent-blue)", border: "1px solid rgba(79,110,247,0.3)" }}>
                      📖 {topic.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Roadmap History */}
            <div className="glass-card p-6 mb-8">
              <h3 className="font-bold mb-4">📜 Roadmap History</h3>
              {roadmapHistory.length > 0 ? (
                <div className="space-y-4">
                  {roadmapHistory.map((entry, i) => {
                    const rmData = entry.roadmap_data;
                    const title = rmData?.title || rmData?.weeks?.[0]?.title || "Learning Roadmap";
                    const totalWeeks = rmData?.weeks?.length || 0;
                    return (
                      <div key={i} className="p-4 rounded-xl" style={{ background: "var(--bg-primary)", border: "1px solid var(--border)" }}>
                        <div className="flex items-center justify-between mb-2">
                          <div>
                            <h4 className="text-sm font-bold">{title}</h4>
                            <p className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
                              {totalWeeks} weeks • Archived {entry.archived_at ? new Date(entry.archived_at).toLocaleDateString() : ""}
                            </p>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold" style={{
                              color: (entry.completion_percentage || 0) >= 80 ? "var(--accent-green)"
                                : (entry.completion_percentage || 0) >= 40 ? "var(--accent-amber)"
                                : "var(--accent-red)"
                            }}>
                              {Math.round(entry.completion_percentage || 0)}%
                            </div>
                            <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>completed</div>
                          </div>
                        </div>
                        <div className="progress-bar h-1.5">
                          <div className="progress-fill" style={{ width: `${entry.completion_percentage || 0}%` }} />
                        </div>
                        {entry.topics_covered && entry.topics_covered.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {entry.topics_covered.slice(0, 6).map((t: string, j: number) => (
                              <span key={j} className="px-2 py-0.5 rounded text-[9px]" style={{ background: "rgba(139,92,246,0.1)", color: "var(--accent-purple)" }}>
                                {t.replace(/_/g, " ")}
                              </span>
                            ))}
                            {entry.topics_covered.length > 6 && (
                              <span className="px-2 py-0.5 rounded text-[9px]" style={{ color: "var(--text-secondary)" }}>
                                +{entry.topics_covered.length - 6} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-3xl mb-2 opacity-50">📜</div>
                  <p className="text-sm" style={{ color: "var(--text-secondary)" }}>No previous roadmaps yet. Complete or quit a roadmap to see it here.</p>
                </div>
              )}
            </div>

            {/* Current Progress */}
            {progressData?.current && (
              <div className="glass-card p-6">
                <h3 className="font-bold mb-4">📈 Current Progress</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                    <div className="text-2xl font-bold gradient-text">{progressData.current.topics_completed?.length || 0}</div>
                    <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Topics Done</div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                    <div className="text-2xl font-bold gradient-text">{progressData.current.topics_studied?.length || 0}</div>
                    <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Topics Studied</div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                    <div className="text-2xl font-bold gradient-text">🔥 {progressData.current.quiz_streak || 0}</div>
                    <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Quiz Streak</div>
                  </div>
                  <div className="text-center p-3 rounded-lg" style={{ background: "var(--bg-primary)" }}>
                    <div className="text-2xl font-bold gradient-text">{progressData.current.badges_earned || 0}</div>
                    <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Badges</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
