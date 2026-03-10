const API_BASE = '/api/v1';

let token: string | null = null;

export function setToken(t: string) {
  token = t;
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem('alvinai_token', t);
  }
}

export function getToken(): string | null {
  if (token) return token;
  if (typeof localStorage !== 'undefined') {
    token = localStorage.getItem('alvinai_token');
  }
  return token;
}

export function clearToken() {
  token = null;
  if (typeof localStorage !== 'undefined') {
    localStorage.removeItem('alvinai_token');
  }
}

async function apiFetch(path: string, options: RequestInit = {}) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };
  const t = getToken();
  if (t) headers['Authorization'] = `Bearer ${t}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (res.status === 401) {
    clearToken();
    if (typeof window !== 'undefined') window.location.href = '/login';
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

export async function login(email: string, password: string) {
  const data = await apiFetch('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  setToken(data.access_token);
  return data;
}

export async function register(email: string, password: string, full_name: string = '') {
  const data = await apiFetch('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, full_name }),
  });
  setToken(data.access_token);
  return data;
}

export async function sendQuery(query: string, namespace: string, conversation_id?: string) {
  return apiFetch('/query/', {
    method: 'POST',
    body: JSON.stringify({ query, namespace, conversation_id }),
  });
}

export async function getConversations() {
  return apiFetch('/conversations/');
}

export async function getConversation(id: string) {
  return apiFetch(`/conversations/${id}`);
}

export async function healthCheck() {
  return apiFetch('/health');
}
