/**
 * EduPath AI — Topic Study Page
 * Team KRIYA | Meta Hackathon 2026
 *
 * Per-topic learning page with AI-generated summary, ranked course
 * cards from real platforms, alternative course discovery, downloadable
 * resources, and mark-as-complete workflow that unlocks quizzes.
 */

"use client";

import { useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Suspense } from "react";
import Link from "next/link";
import { apiGet, apiPost } from "@/lib/api";
import { SOURCE_CONFIG } from "@/lib/resourceUtils";

interface ResourceCard {
  title: string;
  url: string;
  source: string;
  description: string;
  duration_estimate: string;
  resource_type: string;
}

interface TopicData {
  topic_id: string;
  topic_name: string;
  ai_summary: string;
  resources: ResourceCard[];
  can_mark_complete: boolean;
  quiz_unlocked: boolean;
}

function TopicContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const topicId = searchParams.get("id") as string;
  
  const [topicData, setTopicData] = useState<TopicData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [canMarkComplete, setCanMarkComplete] = useState(false);
  const [markingComplete, setMarkingComplete] = useState(false);

  // Alternative courses state
  const [altCourses, setAltCourses] = useState<any[]>([]);
  const [loadingAlt, setLoadingAlt] = useState(false);
  const [showAlt, setShowAlt] = useState(false);
  const [noMoreCourses, setNoMoreCourses] = useState(false);

  useEffect(() => {
    const fetchTopic = async () => {
      try {
        const studentId = localStorage.getItem("edupath_student_id") || "";
        const data = await apiGet(`/resources/${topicId}?student_id=${studentId}`);
        setTopicData(data);
        setCanMarkComplete(data.can_mark_complete);
      } catch (err: any) {
        setError(err.message || "Failed to load topic resources");
      } finally {
        setLoading(false);
      }
    };
    
    if (topicId) {
      fetchTopic();
    }
  }, [topicId]);

  const handleResourceClick = async (url: string) => {
    window.open(url, "_blank");

    const studentId = localStorage.getItem("edupath_student_id");
    if (!studentId) return;

    try {
      const data = await apiPost(`/resources/${topicId}/link-clicked`, {
        student_id: studentId,
        resource_url: url
      });
      if (data.can_mark_complete) {
        setCanMarkComplete(true);
      }
    } catch (err) {
      console.error("Failed to record link click:", err);
    }
  };

  const handleMarkComplete = async () => {
    const studentId = localStorage.getItem("edupath_student_id");
    if (!studentId) return;

    setMarkingComplete(true);
    try {
      const data = await apiPost(`/resources/${topicId}/mark-complete`, {
        student_id: studentId
      });
      
      if (data.quiz_unlocked) {
        setTopicData(prev => prev ? { ...prev, quiz_unlocked: true } : prev);
      }
    } catch (err: any) {
      alert(err.message || "Failed to mark complete. Did you click a resource link first?");
    } finally {
      setMarkingComplete(false);
    }
  };

  const handleTryAnotherCourse = async () => {
    setLoadingAlt(true);
    try {
      const data = await apiGet(`/resources/${topicId}/alternative`);
      if (data.resources && data.resources.length > 0) {
        setAltCourses(data.resources);
        setShowAlt(true);
        setNoMoreCourses(!data.has_more);
      } else {
        setNoMoreCourses(true);
      }
    } catch (err) {
      console.error("Failed to load alternative courses:", err);
      setNoMoreCourses(true);
    } finally {
      setLoadingAlt(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen p-8" style={{ background: "var(--bg-primary)" }}>
        <Link href="/dashboard" className="text-sm mb-6 inline-block" style={{ color: "var(--text-secondary)" }}>← Back to Dashboard</Link>
        <div className="glass-card p-8 animate-pulse">
          <div className="h-8 bg-gray-700 w-1/3 rounded mb-4"></div>
          <div className="h-4 bg-gray-700 w-1/4 rounded mb-8"></div>
          <div className="h-32 bg-gray-700 w-full rounded mb-8"></div>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="h-24 bg-gray-700 rounded"></div>
            <div className="h-24 bg-gray-700 rounded"></div>
            <div className="h-24 bg-gray-700 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen p-8" style={{ background: "var(--bg-primary)" }}>
        <Link href="/dashboard" className="text-sm mb-6 inline-block" style={{ color: "var(--text-secondary)" }}>← Back to Dashboard</Link>
        <div className="glass-card p-8 text-center text-red-400">
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!topicData) return null;

  const renderCourseCard = (res: any, i: number, isFirst: boolean = false) => {
    const config = SOURCE_CONFIG[res.source] || SOURCE_CONFIG["Other"] || { color: "#6B7280", label: res.source || "Resource" };
    const isFallback = res.is_fallback;
    return (
      <div key={i} className="glass-card p-5 flex flex-col h-full hover:scale-[1.02] transition-transform relative">
        {/* Top Pick badge for first result */}
        {isFirst && !isFallback && (
          <div className="absolute -top-2 -right-2 px-2 py-0.5 rounded-full text-[9px] font-bold z-10"
            style={{ background: "linear-gradient(135deg, var(--accent-blue), var(--accent-purple))", color: "white" }}>
            🏆 Top Pick
          </div>
        )}

        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-bold px-2 py-1 rounded" style={{ backgroundColor: `${config.color}20`, color: config.color }}>
            {config.label}
          </span>
          <div className="flex items-center gap-2">
            {res.rating && (
              <span className="text-xs font-bold" style={{ color: "var(--accent-amber)" }}>
                ⭐ {res.rating.toFixed(1)}
              </span>
            )}
            <span className="text-xs opacity-50">{res.duration_estimate}</span>
          </div>
        </div>
        <h3 className="font-bold text-sm mb-2 line-clamp-2">{res.title}</h3>
        <p className="text-xs opacity-70 mb-2 flex-grow line-clamp-3">{res.description}</p>
        
        {/* AI recommendation reason */}
        {res.ai_reason && (
          <div className="text-[10px] mb-3 px-2 py-1 rounded-lg" style={{ background: "rgba(139,92,246,0.1)", color: "var(--accent-purple)" }}>
            🤖 {res.ai_reason}
          </div>
        )}
        
        <button 
          onClick={() => handleResourceClick(res.url)}
          className="w-full text-center py-2 rounded-lg text-sm font-bold transition-all mt-auto"
          style={{ background: `${config.color}15`, color: config.color, border: `1px solid ${config.color}30` }}
        >
          {isFallback ? "🔍 Search on Google →" : "Go to Course →"}
        </button>
      </div>
    );
  };

  return (
    <div className="min-h-screen p-6 md:p-12" style={{ background: "var(--bg-primary)" }}>
      <Link href="/dashboard" className="text-sm mb-6 inline-block hover:underline" style={{ color: "var(--text-secondary)" }}>
        ← Back to Roadmap
      </Link>

      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Topic: {topicData.topic_name}</h1>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          AI-ranked courses from top platforms. Best matches shown first.
        </p>
      </div>

      <div className="glass-card p-6 md:p-8 mb-8">
        <h2 className="text-sm font-bold opacity-50 mb-4 tracking-wider">AI SUMMARY</h2>
        <div className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          {topicData.ai_summary.split('\n\n').map((paragraph, i) => (
            <p key={i} className="mb-4">{paragraph}</p>
          ))}
        </div>
      </div>

      <h2 className="text-sm font-bold opacity-50 mb-4 tracking-wider">🏅 TOP COURSES & RESOURCES</h2>
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
        {topicData.resources.map((res, i) => renderCourseCard(res, i, i === 0))}
      </div>

      {/* Try Another Course button */}
      <div className="flex items-center gap-3 mb-8">
        <button
          onClick={handleTryAnotherCourse}
          disabled={loadingAlt || (showAlt && noMoreCourses)}
          className="px-5 py-2.5 rounded-xl text-sm font-bold flex items-center gap-2 transition-all hover:scale-105 disabled:opacity-50"
          style={{ background: "rgba(139,92,246,0.15)", color: "var(--accent-purple)", border: "1px solid rgba(139,92,246,0.3)" }}
        >
          {loadingAlt ? (
            <>
              <span className="animate-spin">↻</span> Finding more courses...
            </>
          ) : showAlt && noMoreCourses ? (
            "No more courses available"
          ) : (
            <>🔄 Try Another Course</>
          )}
        </button>

        {showAlt && noMoreCourses && (
          <a
            href={`https://www.google.com/search?q=${topicData.topic_name.replace(/ /g, '+')}+best+free+course+${new Date().getFullYear()}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-5 py-2.5 rounded-xl text-sm font-bold flex items-center gap-2 transition-all hover:scale-105"
            style={{ background: "rgba(79,110,247,0.15)", color: "var(--accent-blue)", border: "1px solid rgba(79,110,247,0.3)" }}
          >
            🔍 Search on Google
          </a>
        )}
      </div>

      {/* Alternative courses section */}
      {showAlt && altCourses.length > 0 && (
        <div className="mb-8">
          <h2 className="text-sm font-bold opacity-50 mb-4 tracking-wider">🔄 MORE COURSES</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {altCourses.map((res, i) => renderCourseCard(res, i + 10, false))}
          </div>
        </div>
      )}

      {/* ── Download Resources Section ── */}
      <h2 className="text-sm font-bold opacity-50 mb-4 tracking-wider">📥 DOWNLOAD NOTES & RESOURCES</h2>
      <div className="grid md:grid-cols-3 gap-4 mb-8">
        <a href={`https://www.google.com/search?q=${topicData.topic_name.replace(/ /g, '+')}+cheat+sheet+pdf`}
          target="_blank" rel="noopener noreferrer"
          className="glass-card p-5 text-center hover:scale-[1.02] transition-transform block">
          <div className="text-2xl mb-2">📄</div>
          <h3 className="font-bold text-sm mb-1">Cheat Sheet</h3>
          <p className="text-[10px] opacity-60">Quick reference PDF for {topicData.topic_name}</p>
        </a>
        <a href={`https://www.google.com/search?q=${topicData.topic_name.replace(/ /g, '+')}+study+notes+pdf+free`}
          target="_blank" rel="noopener noreferrer"
          className="glass-card p-5 text-center hover:scale-[1.02] transition-transform block">
          <div className="text-2xl mb-2">📝</div>
          <h3 className="font-bold text-sm mb-1">Study Notes</h3>
          <p className="text-[10px] opacity-60">Detailed notes & summaries to refer later</p>
        </a>
        <a href={`https://www.google.com/search?q=${topicData.topic_name.replace(/ /g, '+')}+practice+exercises+free`}
          target="_blank" rel="noopener noreferrer"
          className="glass-card p-5 text-center hover:scale-[1.02] transition-transform block">
          <div className="text-2xl mb-2">🏋️</div>
          <h3 className="font-bold text-sm mb-1">Practice Exercises</h3>
          <p className="text-[10px] opacity-60">Hands-on exercises to test your skills</p>
        </a>
      </div>

      {/* ── Mark Complete & Quiz Section ── */}
      <div className="glass-card p-6 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <h3 className="font-bold">Ready to take the quiz?</h3>
          <p className="text-xs opacity-70">Mark this topic as complete to unlock the quiz on your dashboard.</p>
        </div>
        
        <div className="flex items-center gap-3">
          {topicData.quiz_unlocked ? (
            <>
              <button disabled className="px-6 py-3 rounded-lg text-sm font-bold whitespace-nowrap transition-all bg-green-900 text-green-300 opacity-80 cursor-not-allowed border border-green-500">
                Completed ✓
              </button>
              <Link href="/dashboard" className="px-6 py-3 rounded-lg text-sm font-bold whitespace-nowrap transition-all" style={{ background: "transparent", color: "var(--accent-blue)", border: "1px solid var(--accent-blue)" }}>
                Take Quiz →
              </Link>
            </>
          ) : (
            <button 
              onClick={handleMarkComplete}
              disabled={!canMarkComplete || markingComplete}
              className={`px-6 py-3 rounded-lg text-sm font-bold whitespace-nowrap transition-all ${
                canMarkComplete && !markingComplete ? 'glow-btn' : 'opacity-40 cursor-not-allowed bg-gray-800'
              }`}
            >
              {markingComplete ? "Marking..." : "Mark as Complete ✓"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default function TopicPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen p-8 text-center" style={{ background: "var(--bg-primary)", color: "var(--text-secondary)" }}>
        Loading Topic...
      </div>
    }>
      <TopicContent />
    </Suspense>
  );
}
