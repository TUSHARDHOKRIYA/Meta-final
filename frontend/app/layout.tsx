/**
 * EduPath AI — Root Layout
 * Team KRIYA | Meta Hackathon 2026
 *
 * Next.js App Router root layout. Sets global font (Inter), dark mode,
 * and SEO metadata for the entire application.
 */

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "EduPath AI — Personalized Learning Tutor",
  description: "AI-powered personalized learning roadmaps for any field. Adaptive quizzes, projects, badges, and career preparation.",
  keywords: ["AI tutor", "personalized learning", "education", "roadmap", "adaptive quiz"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} antialiased`}>
        {children}
      </body>
    </html>
  );
}
