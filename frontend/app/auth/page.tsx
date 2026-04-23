/**
 * EduPath AI — Authentication Page
 * Team KRIYA | Meta Hackathon 2026
 *
 * Login / Signup page with email+password and Google OAuth via
 * Supabase Auth. Redirects to dashboard on success.
 */

"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";

export default function AuthPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  // Check if already logged in
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) {
        router.push("/dashboard");
      }
    });
  }, [router]);

  const handleLogin = async () => {
    setLoading(true);
    setError("");
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      setError(error.message);
    } else {
      router.push("/dashboard");
    }
    setLoading(false);
  };

  const handleSignup = async () => {
    setLoading(true);
    setError("");
    setMessage("");

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { full_name: name },
        emailRedirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error) {
      setError(error.message);
    } else if (data.user && !data.session) {
      // Email confirmation required
      setMessage("Check your email for a confirmation link! Then come back and log in.");
      setMode("login");
    } else {
      // Auto-confirmed, redirect to onboarding
      router.push("/onboarding");
    }
    setLoading(false);
  };

  const handleGoogleLogin = async () => {
    setLoading(true);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    });
    if (error) setError(error.message);
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ background: "var(--bg-primary)" }}>
      {/* Background glow */}
      <div className="fixed top-20 left-1/2 -translate-x-1/2 w-[500px] h-[500px] rounded-full opacity-15" style={{ background: "radial-gradient(circle, var(--accent-blue), transparent 70%)" }} />

      <div className="relative z-10 w-full max-w-md">
        <Link href="/" className="inline-flex items-center gap-2 mb-8 text-sm transition-colors" style={{ color: "var(--text-secondary)" }}>
          ← Back to Home
        </Link>

        <div className="glass-card p-8">
          <div className="text-center mb-6">
            <div className="text-4xl mb-2">🎓</div>
            <h1 className="text-2xl font-bold gradient-text">
              {mode === "login" ? "Welcome Back" : "Create Account"}
            </h1>
            <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>
              {mode === "login" ? "Log in to continue your learning journey" : "Start your personalized learning path"}
            </p>
          </div>

          {/* Toggle */}
          <div className="flex rounded-xl overflow-hidden mb-6" style={{ background: "var(--bg-primary)", border: "1px solid var(--border)" }}>
            <button onClick={() => setMode("login")}
              className="flex-1 py-2.5 text-sm font-bold transition-all"
              style={{ background: mode === "login" ? "rgba(79,110,247,0.2)" : "transparent", color: mode === "login" ? "var(--accent-blue)" : "var(--text-secondary)" }}>
              Log In
            </button>
            <button onClick={() => setMode("signup")}
              className="flex-1 py-2.5 text-sm font-bold transition-all"
              style={{ background: mode === "signup" ? "rgba(79,110,247,0.2)" : "transparent", color: mode === "signup" ? "var(--accent-blue)" : "var(--text-secondary)" }}>
              Sign Up
            </button>
          </div>

          {/* Google Login */}
          <button onClick={handleGoogleLogin} disabled={loading}
            className="w-full py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 mb-4 transition-all hover:opacity-90"
            style={{ background: "var(--bg-primary)", border: "1px solid var(--border)" }}>
            <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
            Continue with Google
          </button>

          <div className="flex items-center gap-3 mb-4">
            <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>or</span>
            <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
          </div>

          {/* Form */}
          <div className="space-y-3">
            {mode === "signup" && (
              <input type="text" value={name} onChange={(e) => setName(e.target.value)}
                className="w-full px-4 py-3 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 transition-all text-sm"
                style={{ background: "var(--bg-primary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                placeholder="Full Name" />
            )}
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 transition-all text-sm"
              style={{ background: "var(--bg-primary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
              placeholder="Email address" />
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (mode === "login" ? handleLogin() : handleSignup())}
              className="w-full px-4 py-3 rounded-xl outline-none focus:ring-2 focus:ring-blue-500 transition-all text-sm"
              style={{ background: "var(--bg-primary)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
              placeholder="Password (min 6 characters)" />
          </div>

          {error && <p className="text-sm mt-3 px-3 py-2 rounded-lg" style={{ background: "rgba(239,68,68,0.1)", color: "var(--accent-red)" }}>{error}</p>}
          {message && <p className="text-sm mt-3 px-3 py-2 rounded-lg" style={{ background: "rgba(34,197,94,0.1)", color: "var(--accent-green)" }}>{message}</p>}

          <button onClick={mode === "login" ? handleLogin : handleSignup}
            disabled={loading || !email || !password || (mode === "signup" && !name)}
            className="glow-btn w-full mt-4 disabled:opacity-50">
            {loading ? "Processing..." : mode === "login" ? "Log In →" : "Create Account →"}
          </button>
        </div>
      </div>
    </div>
  );
}
