/**
 * EduPath AI — Resource Display Utilities
 * Team KRIYA | Meta Hackathon 2026
 *
 * Maps learning-platform names to brand colours and labels for
 * rendering resource cards in the frontend dashboard.
 */

export const SOURCE_CONFIG: Record<string, { color: string; label: string }> = {
  "Kaggle":       { color: "#20BEFF", label: "Kaggle Learn"    },
  "freeCodeCamp": { color: "#0A0A23", label: "freeCodeCamp"    },
  "fast.ai":      { color: "#FF6B6B", label: "fast.ai"         },
  "HuggingFace":  { color: "#FFD21E", label: "HuggingFace"     },
  "MIT OCW":      { color: "#A31F34", label: "MIT OpenCourseWare" },
  "Coursera":     { color: "#0056D2", label: "Coursera"        },
  "edX":          { color: "#02262B", label: "edX"             },
  "Khan Academy": { color: "#14BF96", label: "Khan Academy"    },
  "Udemy":        { color: "#A435F0", label: "Udemy"           },
  "YouTube":      { color: "#FF0000", label: "YouTube"         },
  "Codecademy":   { color: "#1F4056", label: "Codecademy"      },
  "Pluralsight":  { color: "#E80A89", label: "Pluralsight"     },
  "Google":       { color: "#4285F4", label: "Google Search"   },
  "Other":        { color: "#6B7280", label: "Free Resource"   },
};
