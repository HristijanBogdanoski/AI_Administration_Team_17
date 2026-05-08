import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Footer from "../components/Footer";

const ROLE_LABELS = { admin: "Админ", user: "Корисник" };
const ROLE_COLORS = { admin: "#1B3A6B", user: "#6b7280" };

function getTokenPayload() {
    const token = localStorage.getItem("token");
    if (!token) return null;
    try { return JSON.parse(atob(token.split(".")[1])); } catch { return null; }
}

export default function AdminUsers() {
    const navigate = useNavigate();
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [confirmDelete, setConfirmDelete] = useState(null);
    const [roleLoading, setRoleLoading] = useState(null);

    const payload = getTokenPayload();
    const currentEmail = payload?.sub;

    useEffect(() => {
        if (!payload || payload.role !== "admin") {
            navigate("/");
            return;
        }
        fetchUsers();
    }, []);

    const fetchUsers = async () => {
        const token = localStorage.getItem("token");
        try {
            setLoading(true);
            const res = await fetch("http://127.0.0.1:8000/auth/admin/users", {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!res.ok) throw new Error();
            const data = await res.json();
            setUsers(data);
        } catch {
            setError("Грешка при вчитување на корисници.");
        } finally {
            setLoading(false);
        }
    };

    const handleRoleChange = async (email, newRole) => {
        const token = localStorage.getItem("token");
        setRoleLoading(email);
        try {
            const res = await fetch(
                `http://127.0.0.1:8000/auth/admin/users/${encodeURIComponent(email)}/role?role=${newRole}`,
                { method: "PUT", headers: { Authorization: `Bearer ${token}` } }
            );
            if (!res.ok) throw new Error();
            const updated = await res.json();
            setUsers((prev) => prev.map((u) => (u.email === email ? updated : u)));
        } catch {
            setError("Грешка при промена на улога.");
        } finally {
            setRoleLoading(null);
        }
    };

    const handleDelete = async (email) => {
        const token = localStorage.getItem("token");
        try {
            const res = await fetch(
                `http://127.0.0.1:8000/auth/admin/users/${encodeURIComponent(email)}`,
                { method: "DELETE", headers: { Authorization: `Bearer ${token}` } }
            );
            if (!res.ok) throw new Error();
            setUsers((prev) => prev.filter((u) => u.email !== email));
            setConfirmDelete(null);
        } catch {
            setError("Грешка при бришење на корисник.");
        }
    };

    return (
        <div style={{ backgroundColor: "#f8fafc", minHeight: "100vh" }}>
            <div style={{ background: "linear-gradient(150deg, #0f2044 0%, #1B3A6B 55%, #1a4a8a 100%)", padding: "52px 40px 68px", textAlign: "center", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse 60% 70% at 50% 120%, rgba(212,160,23,0.1) 0%, transparent 70%)" }} />
                <div style={{ width: 64, height: 64, background: "rgba(255,255,255,0.1)", backdropFilter: "blur(12px)", borderRadius: 18, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 22px", border: "1px solid rgba(255,255,255,0.15)", position: "relative", zIndex: 2 }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgb(212,160,23)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                    </svg>
                </div>
                <h1 style={{ color: "#fff", fontSize: "2.2rem", fontWeight: 800, margin: "0 0 10px", position: "relative", zIndex: 2 }}>Корисници</h1>
                <p style={{ color: "#93c5fd", fontSize: "0.92rem", margin: 0, position: "relative", zIndex: 2 }}>Управување со корисници и улоги</p>
            </div>

            <div style={{ maxWidth: 900, margin: "40px auto 80px", padding: "0 20px" }}>
                {error && (
                    <div style={{ padding: "12px 16px", backgroundColor: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, color: "#dc2626", marginBottom: 20 }}>
                        {error}
                    </div>
                )}

                {loading ? (
                    <p style={{ textAlign: "center", color: "#64748b" }}>Се вчитуваат корисници...</p>
                ) : (
                    <div style={{ backgroundColor: "#fff", borderRadius: 16, border: "1px solid #e2e8f0", overflow: "hidden" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse" }}>
                            <thead>
                                <tr style={{ backgroundColor: "#f1f5f9" }}>
                                    <th style={{ padding: "14px 20px", textAlign: "left", fontSize: "0.8rem", color: "#64748b", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>Корисник</th>
                                    <th style={{ padding: "14px 20px", textAlign: "left", fontSize: "0.8rem", color: "#64748b", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>ЕМБГ</th>
                                    <th style={{ padding: "14px 20px", textAlign: "left", fontSize: "0.8rem", color: "#64748b", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>Улога</th>
                                    <th style={{ padding: "14px 20px", textAlign: "left", fontSize: "0.8rem", color: "#64748b", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>Акции</th>
                                </tr>
                            </thead>
                            <tbody>
                                {users.map((user, i) => (
                                    <tr key={user.id} style={{ borderTop: "1px solid #f1f5f9", backgroundColor: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                                        <td style={{ padding: "16px 20px" }}>
                                            <div style={{ fontWeight: 600, color: "#1e293b", fontSize: "0.95rem" }}>{user.full_name || "—"}</div>
                                            <div style={{ color: "#64748b", fontSize: "0.82rem" }}>{user.email}</div>
                                        </td>
                                        <td style={{ padding: "16px 20px", color: "#64748b", fontSize: "0.88rem" }}>
                                            {user.embg || "—"}
                                        </td>
                                        <td style={{ padding: "16px 20px" }}>
                                            <span style={{ backgroundColor: `${ROLE_COLORS[user.role]}15`, color: ROLE_COLORS[user.role], padding: "4px 10px", borderRadius: 6, fontSize: "0.8rem", fontWeight: 700 }}>
                                                {ROLE_LABELS[user.role] || user.role}
                                            </span>
                                        </td>
                                        <td style={{ padding: "16px 20px" }}>
                                            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                                                {user.email !== currentEmail ? (
                                                    <>
                                                        <select
                                                            value={user.role}
                                                            disabled={roleLoading === user.email}
                                                            onChange={(e) => handleRoleChange(user.email, e.target.value)}
                                                            style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #e2e8f0", fontSize: "0.82rem", cursor: "pointer", backgroundColor: "#fff" }}
                                                        >
                                                            <option value="user">Корисник</option>
                                                            <option value="admin">Админ</option>
                                                        </select>
                                                        <button
                                                            onClick={() => setConfirmDelete(user)}
                                                            style={{ backgroundColor: "#ef4444", color: "#fff", border: "none", padding: "6px 12px", borderRadius: 6, cursor: "pointer", fontSize: "0.82rem", fontWeight: 600 }}
                                                        >
                                                            Избриши
                                                        </button>
                                                    </>
                                                ) : (
                                                    <span style={{ color: "#94a3b8", fontSize: "0.82rem" }}>Тоа сте вие</span>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {confirmDelete && (
                <div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.6)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 1000 }} onClick={() => setConfirmDelete(null)}>
                    <div style={{ backgroundColor: "#fff", padding: 30, borderRadius: 12, width: "100%", maxWidth: 400 }} onClick={(e) => e.stopPropagation()}>
                        <h3 style={{ color: "#dc2626", marginBottom: 16 }}>Избриши корисник</h3>
                        <p style={{ color: "#6b7280", marginBottom: 24 }}>
                            Дали сте сигурни дека сакате да го избришете корисникот <strong>{confirmDelete.email}</strong>?
                        </p>
                        <div style={{ display: "flex", gap: 10 }}>
                            <button onClick={() => handleDelete(confirmDelete.email)} style={{ flex: 1, padding: 10, backgroundColor: "#dc2626", color: "#fff", border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 600 }}>Избриши</button>
                            <button onClick={() => setConfirmDelete(null)} style={{ flex: 1, padding: 10, backgroundColor: "#6b7280", color: "#fff", border: "none", borderRadius: 6, cursor: "pointer" }}>Откажи</button>
                        </div>
                    </div>
                </div>
            )}

            <Footer />
        </div>
    );
}