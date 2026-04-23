/**
 * EduPath AI — Landing Page
 * Team KRIYA | Meta Hackathon 2026
 *
 * Marketing homepage showcasing platform features, supported fields,
 * and the 4-step onboarding flow. Includes animated field carousel
 * and glassmorphism UI.
 */

"use client";

import Link from "next/link";
import { useState, useEffect } from "react";

const FEATURES = [
  { icon: "🎯", title: "Smart Onboarding", desc: "2-minute setup with resume parsing and skill assessment" },
  { icon: "🗺️", title: "AI Roadmaps", desc: "Personalized learning paths for any field — tech, healthcare, law, business" },
  { icon: "📝", title: "Adaptive Quizzes", desc: "AI-generated quizzes that adapt to your performance" },
  { icon: "🛠️", title: "Project Milestones", desc: "Field-specific mini projects and capstones to reinforce learning" },
  { icon: "🏆", title: "Badges & Streaks", desc: "Achievement system with transparent, deterministic criteria" },
  { icon: "💼", title: "Job Readiness", desc: "Career score, interview prep, hackathons & events" },
];

const FIELDS = ["Tech", "Healthcare", "Business", "Law", "Design", "Any Field"];

export default function Home() {
  const [activeField, setActiveField] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveField((prev) => (prev + 1) % FIELDS.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-primary)" }}>
      {/* ── Navbar ─── */}
      <nav className="fixed top-0 w-full z-50 px-6 py-4 flex items-center justify-between" style={{ background: "rgba(10,10,15,0.8)", backdropFilter: "blur(20px)", borderBottom: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2">
          <span className="text-2xl">🎓</span>
          <span className="text-xl font-bold gradient-text">EduPath AI</span>
        </div>
        <div className="flex items-center gap-6">
          <a href="#features" className="text-sm hover:text-white transition-colors" style={{ color: "var(--text-secondary)" }}>Features</a>
          <a href="#how-it-works" className="text-sm hover:text-white transition-colors" style={{ color: "var(--text-secondary)" }}>How It Works</a>
          <Link href="/auth" className="glow-btn text-sm !py-2 !px-6">
            Get Started →
          </Link>
        </div>
      </nav>

      {/* ── Hero Section ─── */}
      <section className="pt-32 pb-20 px-6 text-center relative overflow-hidden">
        {/* Decorative bg glow */}
        <div className="absolute top-20 left-1/2 -translate-x-1/2 w-[600px] h-[600px] rounded-full opacity-20" style={{ background: "radial-gradient(circle, var(--accent-blue), transparent 70%)" }} />

        <div className="relative z-10 max-w-4xl mx-auto">
          <div className="inline-block mb-4 px-4 py-1 rounded-full text-xs font-medium" style={{ background: "rgba(79,110,247,0.15)", color: "var(--accent-blue)", border: "1px solid rgba(79,110,247,0.3)" }}>
            🚀 OpenEnv Hackathon Project
          </div>

          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight animate-fade-in">
            Your AI Tutor for{" "}
            <span className="gradient-text">{FIELDS[activeField]}</span>
          </h1>

          <p className="text-lg md:text-xl mb-10 max-w-2xl mx-auto" style={{ color: "var(--text-secondary)" }}>
            Personalized learning roadmaps that adapt in real-time. Whether you&apos;re a doctor learning AI, an architect exploring BIM, or a student mastering ML — EduPath AI creates your unique path.
          </p>

          <div className="flex gap-4 justify-center">
            <Link href="/auth" className="glow-btn text-lg">
              Start Learning →
            </Link>
            <Link href="/dashboard" className="px-8 py-3 rounded-xl font-bold border transition-all hover:bg-opacity-10" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>
              View Dashboard
            </Link>
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-3 gap-8 max-w-xl mx-auto">
            {[
              { num: "30+", label: "Topics" },
              { num: "5", label: "Fields" },
              { num: "∞", label: "Possibilities" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl font-bold gradient-text">{stat.num}</div>
                <div className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features Grid ─── */}
      <section id="features" className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">Platform Features</h2>
          <p className="text-center mb-12" style={{ color: "var(--text-secondary)" }}>Everything you need for a complete learning experience</p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map((feature, i) => (
              <div key={i} className="glass-card p-6" style={{ animationDelay: `${i * 0.1}s` }}>
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ─── */}
      <section id="how-it-works" className="py-20 px-6" style={{ background: "var(--bg-secondary)" }}>
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">How It Works</h2>
          <p className="mb-12" style={{ color: "var(--text-secondary)" }}>From zero to job-ready in 4 simple steps</p>

          <div className="grid md:grid-cols-4 gap-6">
            {[
              { step: 1, icon: "📄", title: "Upload Resume", desc: "Optional — AI extracts your skills automatically" },
              { step: 2, icon: "🎚️", title: "Assess Skills", desc: "Tell us what you know with granular skill levels" },
              { step: 3, icon: "💼", title: "Paste Job Desc", desc: "Optional — AI maps skill gaps to your dream role" },
              { step: 4, icon: "⏰", title: "Set Time", desc: "How many hours/week? We set realistic milestones" },
            ].map((item) => (
              <div key={item.step} className="relative">
                <div className="w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4 text-xl step-active">{item.step}</div>
                <div className="text-2xl mb-2">{item.icon}</div>
                <h3 className="font-bold mb-1">{item.title}</h3>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{item.desc}</p>
              </div>
            ))}
          </div>

          <Link href="/auth" className="glow-btn inline-block mt-12 text-lg">
            Begin Your Journey →
          </Link>
        </div>
      </section>

      {/* ── Footer ─── */}
      <footer className="py-10 px-6 text-center text-sm" style={{ color: "var(--text-secondary)", borderTop: "1px solid var(--border)" }}>
        <div className="flex items-center justify-center gap-2 mb-2">
          <span>🎓</span>
          <span className="font-bold gradient-text">EduPath AI</span>
        </div>
        <p>Personalized Learning Tutor Environment — OpenEnv Hackathon 2026</p>
      </footer>
    </div>
  );
}
