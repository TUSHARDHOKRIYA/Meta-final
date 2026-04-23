/**
 * EduPath AI — Auth Callback Handler
 * Team KRIYA | Meta Hackathon 2026
 *
 * Handles Supabase OAuth/email-confirmation redirects. Exchanges auth
 * codes for sessions and routes users to onboarding or dashboard based
 * on account age.
 */

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function AuthCallbackPage() {
  const router = useRouter();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [message, setMessage] = useState("Verifying your email...");

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Supabase automatically picks up the tokens from the URL
        // when detectSessionInUrl is true.  Get the session.
        const { data, error } = await supabase.auth.getSession();

        if (error) {
          setStatus("error");
          setMessage(error.message);
          return;
        }

        if (data.session) {
          setStatus("success");
          setMessage("Email confirmed! Redirecting...");

          // Check if user has completed onboarding (profile exists on backend?)
          // For now, always send new confirmed users to onboarding
          // and returning users to dashboard
          const userId = data.session.user.id;
          const createdAt = new Date(data.session.user.created_at);
          const now = new Date();
          const isNewUser = (now.getTime() - createdAt.getTime()) < 5 * 60 * 1000; // within 5 min

          setTimeout(() => {
            if (isNewUser) {
              router.push("/onboarding");
            } else {
              router.push("/dashboard");
            }
          }, 1500);
        } else {
          // No session yet — might be email confirm flow
          // Try to exchange the code from the URL
          const url = new URL(window.location.href);
          const code = url.searchParams.get("code");

          if (code) {
            const { data: exchangeData, error: exchangeError } =
              await supabase.auth.exchangeCodeForSession(code);

            if (exchangeError) {
              setStatus("error");
              setMessage(exchangeError.message);
              return;
            }

            if (exchangeData.session) {
              setStatus("success");
              setMessage("Email confirmed! Redirecting to onboarding...");
              setTimeout(() => router.push("/onboarding"), 1500);
              return;
            }
          }

          // Check hash fragment (implicit flow fallback)
          const hash = window.location.hash;
          if (hash && hash.includes("access_token")) {
            // The Supabase client should have already picked this up
            // Wait a moment and re-check
            await new Promise((r) => setTimeout(r, 1000));
            const { data: retryData } = await supabase.auth.getSession();
            if (retryData.session) {
              setStatus("success");
              setMessage("Email confirmed! Redirecting...");
              setTimeout(() => router.push("/onboarding"), 1500);
              return;
            }
          }

          setStatus("error");
          setMessage("Could not verify your email. Please try logging in manually.");
        }
      } catch (err: any) {
        setStatus("error");
        setMessage(err.message || "Something went wrong");
      }
    };

    handleCallback();
  }, [router]);

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ background: "var(--bg-primary)" }}>
      <div className="fixed top-20 left-1/2 -translate-x-1/2 w-[500px] h-[500px] rounded-full opacity-15" style={{ background: "radial-gradient(circle, var(--accent-blue), transparent 70%)" }} />

      <div className="relative z-10 glass-card p-10 max-w-md w-full text-center">
        {status === "loading" && (
          <>
            <div className="text-5xl mb-4 animate-float">🔐</div>
            <h1 className="text-2xl font-bold gradient-text mb-2">Verifying Email</h1>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{message}</p>
            <div className="mt-6">
              <div className="w-12 h-12 mx-auto rounded-full border-4 border-t-transparent animate-spin" style={{ borderColor: "var(--accent-blue)", borderTopColor: "transparent" }} />
            </div>
          </>
        )}

        {status === "success" && (
          <>
            <div className="text-5xl mb-4">✅</div>
            <h1 className="text-2xl font-bold mb-2" style={{ color: "var(--accent-green)" }}>Email Confirmed!</h1>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{message}</p>
            <div className="mt-6 progress-bar">
              <div className="progress-fill" style={{ width: "100%", transition: "width 1.5s ease" }} />
            </div>
          </>
        )}

        {status === "error" && (
          <>
            <div className="text-5xl mb-4">⚠️</div>
            <h1 className="text-2xl font-bold mb-2" style={{ color: "var(--accent-red)" }}>Verification Issue</h1>
            <p className="text-sm mb-6" style={{ color: "var(--text-secondary)" }}>{message}</p>
            <div className="flex gap-3 justify-center">
              <button onClick={() => router.push("/auth")} className="glow-btn !text-sm">
                Go to Login →
              </button>
              <button onClick={() => window.location.reload()}
                className="px-6 py-2 rounded-lg text-sm font-bold"
                style={{ border: "1px solid var(--border)", color: "var(--text-secondary)" }}>
                Retry
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
