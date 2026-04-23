/**
 * EduPath AI — Frontend API Client
 * Team KRIYA | Meta Hackathon 2026
 *
 * Typed HTTP helpers (GET/POST) for communicating with the FastAPI backend.
 * Reads the backend URL from NEXT_PUBLIC_API_URL and provides user-friendly
 * error messages when the Hugging Face Space is sleeping or unreachable.
 */

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "https://degree-checker-01-meta-new-space.hf.space").replace(/\/+$/, "");

export async function apiGet<T = any>(path: string): Promise<T> {
  const url = `${API_URL}${path}`;
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `GET ${path} failed: ${res.status}`);
    }
    return res.json();
  } catch (err: any) {
    if (err.message?.includes("Failed to fetch") || err.message?.includes("NetworkError")) {
      throw new Error(
        `Cannot reach the backend at ${API_URL}. Please check that your Hugging Face Space is running and not sleeping.`
      );
    }
    throw err;
  }
}

export async function apiPost<T = any>(path: string, body: object): Promise<T> {
  const url = `${API_URL}${path}`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `POST ${path} failed: ${res.status}`);
    }
    return res.json();
  } catch (err: any) {
    if (err.message?.includes("Failed to fetch") || err.message?.includes("NetworkError")) {
      throw new Error(
        `Cannot reach the backend at ${API_URL}. Please check that your Hugging Face Space is running and not sleeping.`
      );
    }
    throw err;
  }
}
